"""Blockwise FP8 linear for XPU.

Xe2 has no FP8 tensor cores, so weights are stored in FP8 (2x memory saving)
with per-128-block scales and dequantized to BF16 for the GEMM, which runs on
the oneDNN-backed BF16 path (measured ~95 TFLOPS on B60, near XMX peak).
Mirrors ``ltx_kernels.blockwise.linear.BlockwiseFP8Linear`` layout so ltx-core
can dispatch to it on XPU.
"""

from __future__ import annotations

from types import ModuleType

import torch
from torch import nn


def _ext() -> ModuleType:
    import xpu_ltx_kernels._C  # noqa: PLC0415

    return xpu_ltx_kernels._C


def fp8_blockwise_quantize_weights(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a ``[out, in]`` bf16 weight to ``(fp8, scale[out//128, in//128])``.

    Uses torch's cast for bit-exact e4m3 rounding; runs once at load time (not
    the hot path). Matches ``ltx_kernels.blockwise_quantize_weights``.
    """
    block = 128
    out, inn = w.shape
    w32 = w.float().view(out // block, block, inn // block, block)
    absmax = w32.abs().amax(dim=(1, 3))
    scale = 448.0 / absmax.clamp_min(1e-8)
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    # Plain layout: q[o, i] corresponds to w[o, i]; per-block scale from
    # absmax[o//128, i//128]. Broadcast scale over the 128x128 blocks.
    q = (w32 * scale[:, None, :, None]).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn)
    q = q.view(out, inn)
    return q, (1.0 / scale).contiguous()


def blockwise_dequantize(w: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize ``(fp8 [out,in], scale)`` back to bf16."""
    return _ext().fp8_dequantize_blockwise(w, scales)


def _dequantize_activations(x_fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize Q8-path activations ``(fp8 [b,s,h], scales [b*s, h//128])`` to bf16."""
    if x_fp8.dim() == 2:
        rows, h = x_fp8.shape
        b, s = rows, 1
    else:
        b, s, h = x_fp8.shape
    sf = scales.reshape(-1)
    nblocks = h // 128
    sr, sc = scales.stride(0), scales.stride(1)
    idx = torch.arange(b * s, device=scales.device)[:, None] * sr + torch.arange(nblocks, device=scales.device)[None, :] * sc
    scale_t = sf[idx]
    xf = x_fp8.float().reshape(b * s, nblocks, 128)
    return (xf * scale_t[:, :, None]).reshape(b, s, h).to(torch.bfloat16)


class BlockwiseFP8Linear(nn.Module):
    """Linear storing weights as FP8 with per-128x128 block scales; computes in BF16.

    Forward: ESIMD dequantize (fp8+scales -> bf16) then ``F.linear`` on the
    oneDNN BF16 GEMM path. Same weight layout as
    ``ltx_kernels.blockwise.linear.BlockwiseFP8Linear``.
    """

    in_features: int
    out_features: int

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device: torch.device | str | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device))
        self.weight_scale = nn.Parameter(
            torch.empty(out_features // 128, in_features // 128, dtype=torch.float32, device=device)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, transform_weights: bool = True) -> "BlockwiseFP8Linear":
        """Build from an ``nn.Linear``; keeps the same shape and (optionally) quantizes weights."""
        out = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, device=linear.weight.device)
        if transform_weights and linear.weight.dtype == torch.bfloat16:
            wq, ws = fp8_blockwise_quantize_weights(linear.weight.detach())
            out.weight.data.copy_(wq)
            out.weight_scale.data.copy_(ws)
            if linear.bias is not None:
                out.bias.data.copy_(linear.bias.detach().float())
        return out

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, tuple):
            # Q8 path feeds (fp8, scales) activation payloads; dequantize first.
            x = _dequantize_activations(*x)
        w = blockwise_dequantize(self.weight, self.weight_scale)
        out = torch.nn.functional.linear(x, w, self.bias.to(w.dtype) if self.bias is not None else None)
        return out
