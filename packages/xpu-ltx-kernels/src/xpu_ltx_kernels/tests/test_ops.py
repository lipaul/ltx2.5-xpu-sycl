"""Correctness tests for xpu_ltx_kernels ops, against torch references."""

import pytest
import torch
from xpu_ltx_kernels import ops as xops

xpu = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")


def _ref_rms_rope(x, cos, sin, w):
    xn = x.float()
    ms = (xn * xn).sum(-1, keepdim=True) / xn.shape[-1]
    xn = xn * torch.rsqrt(ms)
    if w is not None:
        xn = xn * w.float()
    out = torch.empty_like(xn)
    out[..., 0::2] = -xn[..., 1::2] * sin[..., 0::2] + xn[..., 0::2] * cos[..., 0::2]
    out[..., 1::2] = xn[..., 0::2] * sin[..., 1::2] + xn[..., 1::2] * cos[..., 1::2]
    return out


def _ref_split_rope(x, cos, sin, w):
    b, s, h = x.shape
    n = cos.shape[1]
    head = h // n
    half = head // 2
    xn = x.float()
    ms = (xn * xn).sum(-1, keepdim=True) / h
    xn = xn * torch.rsqrt(ms + 1e-6)
    if w is not None:
        xn = xn * w.float().view(1, 1, h)
    xn = xn.view(b, s, n, head)
    c2 = cos.float().permute(0, 2, 1, 3)
    s2 = sin.float().permute(0, 2, 1, 3)
    first = xn[..., :half]
    second = xn[..., half:]
    out_first = c2 * first - s2 * second
    out_second = c2 * second + s2 * first
    return torch.cat([out_first, out_second], dim=-1).view(b, s, h)


@xpu
def test_fp6_pack_matches_reference():
    torch.manual_seed(0)
    w = torch.randint(0, 256, (8, 1024), dtype=torch.uint8, device="xpu")

    def p6(v):
        return (((v >> 7) & 1) << 5) | (((v >> 4) & 1) << 4) | (v & 0xF)

    packed = xops.fp6_pack_tensor(w)
    m, n = w.shape
    np = n * 3 // 4
    assert packed.shape == (m, np)
    wc = w.cpu()
    for row in range(m):
        for cg in range((n + 3) // 4):
            o = cg * 3
            v = [p6(wc[row, cg * 4 + i]) if cg * 4 + i < n else 0 for i in range(4)]
            if o < np:
                assert packed.cpu()[row, o].item() == ((v[0] << 2) | (v[1] >> 4)) & 0xFF
            if o + 1 < np:
                assert packed.cpu()[row, o + 1].item() == ((v[1] << 4) | (v[2] >> 2)) & 0xFF
            if o + 2 < np:
                assert packed.cpu()[row, o + 2].item() == ((v[2] << 6) | v[3]) & 0xFF


@xpu
def test_fp6_unpack_roundtrip_zeroes_e1_e2():
    torch.manual_seed(0)
    w = torch.randint(0, 256, (8, 2048), dtype=torch.uint8, device="xpu")
    packed = xops.fp6_pack_tensor(w)
    unpacked = xops.fp6_unpack_tensor(packed, w.shape[1])
    assert unpacked.shape == w.shape
    assert torch.equal(unpacked.cpu(), (w.cpu() & ~0x60))


@xpu
@pytest.mark.parametrize("dim", [128, 5120])
@pytest.mark.parametrize("rows", [1, 2, 257])
def test_rms_norm_rope(dim, rows):
    torch.manual_seed(0)
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device="xpu")
    cos = torch.randn(rows, dim, dtype=torch.bfloat16, device="xpu").abs()
    sin = torch.randn(rows, dim, dtype=torch.bfloat16, device="xpu").abs()
    w = torch.randn(dim, dtype=torch.bfloat16, device="xpu").abs() + 1
    ref = _ref_rms_rope(x, cos, sin, w)
    got = xops.rms_norm_rope(x, cos, sin, w).float()
    assert (got - ref).abs().max().item() < 0.15


@xpu
@pytest.mark.parametrize(("n", "head"), [(32, 64), (32, 128), (8, 256)])
def test_rms_norm_split_rope(n, head):
    torch.manual_seed(0)
    b, s = 2, 3
    h = n * head
    x = torch.randn(b, s, h, dtype=torch.bfloat16, device="xpu")
    cos = torch.rand(b, n, s, head // 2, dtype=torch.bfloat16, device="xpu")
    sin = torch.rand(b, n, s, head // 2, dtype=torch.bfloat16, device="xpu")
    w = torch.randn(h, dtype=torch.bfloat16, device="xpu").abs() + 1
    ref = _ref_split_rope(x, cos, sin, w)
    got = xops.rms_norm_split_rope(x, cos, sin, w).float()
    assert (got - ref).abs().max().item() < 0.2


@xpu
def test_out_fp8_casts_like_torch():
    torch.manual_seed(0)
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="xpu")
    cos = torch.randn(4, 128, dtype=torch.bfloat16, device="xpu").abs()
    sin = torch.randn(4, 128, dtype=torch.bfloat16, device="xpu").abs()
    got = xops.rms_norm_rope(x, cos, sin, None, out_fp8=True).float()
    ref = _ref_rms_rope(x, cos, sin, None).float()
    # The kernel rounds fp32->bf16 then fp8; tolerance reflects fp8/bf16 precision.
    assert got.dtype == torch.float32
    assert (got - ref).abs().max().item() < 0.3
