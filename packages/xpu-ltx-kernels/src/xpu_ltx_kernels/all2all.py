"""All2All attention-head redistribution for multi-GPU sequence-parallel
inference on XPU.

Functional port of ``ltx_kernels.all_to_all`` for XPU. The CUDA version uses
IPC zero-copy buffers and custom kernels; here we use torch.distributed
collectives (``all_to_all_single``) which route through the XPU-capable
backend (gloo/oneCCL). Same data layout semantics as the CUDA kernel:

  send_recv_heads(x [B, S, H, D]) -> [B, S_total, H/world_size, D]
    - heads are partitioned: rank r owns global heads [r*hpr : (r+1)*hpr]
    - the token dim becomes the concatenation of every rank's tokens, in rank
      order (sender s's tokens land at offset prefix_rank_tokens[s]).
  gather_heads is the inverse.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _assert_usable() -> None:
    if not dist.is_initialized():
        raise RuntimeError("xpu_ltx_kernels.all2all requires torch.distributed to be initialized")


class All2All:
    """Head redistribution across ranks via torch.distributed.

    API mirrors ``ltx_kernels.All2All`` (CUDA/IPC version); the CUDA-only
    ``num_sms`` / timeout args are accepted and ignored.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        seqlen: int,  # noqa: ARG002
        hidden_dim: int,  # noqa: ARG002
        num_sms: int,  # noqa: ARG002 - CUDA-only
        tensor_dtype: torch.dtype,
        group: dist.ProcessGroup | None = None,
        timeout_seconds: float | None = None,  # noqa: ARG002 - CUDA-only
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.tensor_dtype = tensor_dtype
        self._group_size = world_size

    def _token_split_sizes(self) -> list[int]:
        # Uniform sharding assumption (callers pad up-front). Kept as a method
        # so per-rank token counts could be threaded through later.
        return [1] * self._group_size

    def send_recv_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:  # noqa: ARG002
        """Redistribute heads: ``[B, S, H, D]`` -> ``[B, S_total, H/ws, D]``."""
        ws = self._group_size
        if ws == 1:
            return x
        _assert_usable()
        B, S, H, D = x.shape
        hpr = H // ws
        # head-group g of this rank goes to rank g (all_to_all sends part g -> rank g).
        xg = x.view(B, S, ws, hpr, D).permute(2, 0, 1, 3, 4).contiguous().view(ws, -1)
        chunk = xg.shape[1]
        # output part r = concatenation over senders s of sender s's input part r.
        out = torch.empty(ws, ws * chunk, dtype=x.dtype, device=x.device)
        dist.all_to_all_single(out, xg, group=self.group)
        # out[r] = [sender s's head-group r] over s, each [B*S*hpr*D] -> [ws, B, S, hpr, D]
        res = out.view(ws, ws, B, S, hpr, D)
        # rank r keeps res[r] = [sender, B, S, hpr, D]; token = sender*S + s
        local = res[self.rank].permute(1, 0, 2, 3, 4).reshape(B, ws * S, hpr, D)
        return local

    def gather_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:  # noqa: ARG002
        """Gather heads back: ``[B, S_total, H/ws, D]`` -> ``[B, S, H, D]``."""
        ws = self._group_size
        if ws == 1:
            return x
        _assert_usable()
        B, St, hpr, D = x.shape
        S = St // ws
        H = ws * hpr
        # x[b, sender*S+s, :, :] = this rank's head-group at (sender, s). Send it back.
        xg = x.view(B, ws, S, hpr, D).permute(2, 0, 1, 3, 4).contiguous().view(ws, -1)
        chunk = xg.shape[1]
        out = torch.empty(ws, ws * chunk, dtype=x.dtype, device=x.device)
        dist.all_to_all_single(out, xg, group=self.group)
        # out[g] = [each sender's x part g] = that sender's full H at local tokens
        res = out.view(ws, ws, B, S, hpr, D)  # [g, sender, B, S, hpr, D]
        # rank r now reassembles its original H: heads [r*hpr:(r+1)*hpr] were sent
        # by this rank as part r; reconstruct [B, S, ws, hpr, D].
        gathered = torch.empty(B, S, ws, hpr, D, dtype=x.dtype, device=x.device)
        for g in range(ws):
            # out[g] = senders' part-g = this rank's head-group g at local (sender=s) tokens
            piece = res[g][self.rank]  # [B, S, hpr, D]  (sender dim = this rank's local S)
            gathered[..., g, :, :] = piece
        return gathered.reshape(B, S, H, D)

    def set_rank_tokens(self, rank_num_tokens: list[int]) -> None:
        pass  # uniform sharding; no per-rank token state

    def set_timeout_seconds(self, seconds: float) -> None:
        pass

    def destroy(self) -> None:
        pass
