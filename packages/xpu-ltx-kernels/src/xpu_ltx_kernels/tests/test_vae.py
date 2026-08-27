"""Tests for xpu_ltx_kernels.vae (3D neighborhood attention)."""

import pytest
import torch
from xpu_ltx_kernels import vae

xpu = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")


def _ref_na3d(q, k, v, kernel_size, scale):
    """Eager reference matching natten semantics (clamped windows)."""
    B, T, H, W, NH, _ = q.shape
    kt, kh, kw = kernel_size
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.empty_like(qf)
    for b in range(B):
        for t in range(T):
            t_lo = min(max(t - kt // 2, 0), T - kt)
            for h in range(H):
                h_lo = min(max(h - kh // 2, 0), H - kh)
                for w in range(W):
                    w_lo = min(max(w - kw // 2, 0), W - kw)
                    for nh in range(NH):
                        qrow = qf[b, t, h, w, nh]
                        scores = []
                        vals = []
                        for tk in range(t_lo, t_lo + kt):
                            for hk in range(h_lo, h_lo + kh):
                                for wk in range(w_lo, w_lo + kw):
                                    scores.append((qrow * kf[b, tk, hk, wk, nh]).sum() * scale)
                                    vals.append(vf[b, tk, hk, wk, nh])
                        s = torch.stack(scores)
                        p = torch.softmax(s, dim=0)
                        out[b, t, h, w, nh] = sum(p[i] * vals[i] for i in range(len(vals)))
    return out


@xpu
@pytest.mark.parametrize(("dim", "kernel_size"), [(4, 3), (6, 5)])
def test_na3d_matches_reference(dim, kernel_size):
    torch.manual_seed(0)
    q = torch.randn(1, dim, dim, dim, 2, 8, dtype=torch.bfloat16, device="xpu")
    k = torch.randn(1, dim, dim, dim, 2, 8, dtype=torch.bfloat16, device="xpu")
    v = torch.randn(1, dim, dim, dim, 2, 8, dtype=torch.bfloat16, device="xpu")
    ks = (kernel_size,) * 3
    ref = _ref_na3d(q, k, v, ks, 0.353553)  # 1/sqrt(8)
    got = vae.na3d(q, k, v, ks, 0.353553).float()
    err = (got - ref).abs().max().item()
    assert err < 0.05, err
