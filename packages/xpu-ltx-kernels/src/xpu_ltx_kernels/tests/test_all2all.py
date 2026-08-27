"""Tests for xpu_ltx_kernels.all2all head-redistribution permutation.

The real collective is ``dist.all_to_all_single``; here we validate the
permutation math by substituting a reference all_to_all that operates on the
"global" per-rank inputs, so the roundtrip is checkable in a single process.
"""

import torch
import torch.distributed as dist
import xpu_ltx_kernels.all2all as a2a


def test_send_recv_gather_roundtrip(monkeypatch):
    torch.manual_seed(0)
    ws = 2
    B, S, H, D = 1, 3, 4, 8
    hpr = H // ws
    all_x = torch.arange(ws * B * S * H * D, dtype=torch.float32).reshape(ws, B, S, H, D)
    # global_inputs[r] = rank r's input to all_to_all = its head-groups stacked.
    global_inputs = [
        all_x[r].view(B, S, ws, hpr, D).permute(2, 0, 1, 3, 4).contiguous().view(ws, -1) for r in range(ws)
    ]

    def fake_all_to_all(output, inp, group=None):  # noqa: ARG001
        for r in range(ws):
            output[r].copy_(torch.cat([global_inputs[s][r] for s in range(ws)], 0))

    monkeypatch.setattr(dist, "all_to_all_single", fake_all_to_all)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    for r in range(ws):
        obj = a2a.All2All(rank=r, world_size=ws, seqlen=S, hidden_dim=H, num_sms=0, tensor_dtype=torch.float32)
        x = all_x[r].clone()
        out = obj.send_recv_heads(x)
        assert out.shape == (B, ws * S, hpr, D)
        for sender in range(ws):
            for s in range(S):
                for h in range(hpr):
                    exp = all_x[sender][0, s, r * hpr + h]
                    assert torch.equal(out[0, sender * S + s, h], exp), (r, sender, s, h)
        back = obj.gather_heads(out)
        assert torch.equal(back, x)


def test_single_rank_is_identity():
    obj = a2a.All2All(rank=0, world_size=1, seqlen=4, hidden_dim=8, num_sms=0, tensor_dtype=torch.float32)
    x = torch.randn(1, 4, 8, 8)
    assert torch.equal(obj.send_recv_heads(x), x)
    assert torch.equal(obj.gather_heads(x), x)
