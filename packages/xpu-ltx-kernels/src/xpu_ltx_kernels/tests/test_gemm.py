"""Tests for xpu_ltx_kernels.gemm (blockwise FP8 storage + bf16 GEMM)."""

import pytest
import torch
from xpu_ltx_kernels import gemm

xpu = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")


@xpu
def test_quantize_dequantize_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(256, 512, dtype=torch.bfloat16, device="xpu")
    q, s = gemm.fp8_blockwise_quantize_weights(w)
    assert q.shape == w.shape
    assert q.dtype == torch.float8_e4m3fn
    assert s.shape == (2, 4)
    dq = gemm.blockwise_dequantize(q, s)
    rel = ((dq.float() - w.float()).abs() / w.float().abs().clamp_min(0.05)).max().item()
    assert rel < 0.1


@xpu
def test_quantize_matches_torch_reference():
    torch.manual_seed(0)
    w = torch.randn(256, 512, dtype=torch.bfloat16, device="xpu")
    q, _ = gemm.fp8_blockwise_quantize_weights(w)
    w32 = w.float().view(2, 128, 4, 128)
    absmax = w32.abs().amax(dim=(1, 3))
    scale = 448.0 / absmax
    ref = (
        (w32 * scale[:, None, :, None])
        .clamp(torch.finfo(torch.float8_e4m3fn).min, torch.finfo(torch.float8_e4m3fn).max)
        .to(torch.float8_e4m3fn)
        .view(256, 512)
    )
    assert torch.equal(q.cpu(), ref.cpu())


@xpu
def test_linearxpu_forward_close_to_bf16():
    torch.manual_seed(0)
    ref = torch.nn.Linear(512, 256, bias=True, device="xpu", dtype=torch.bfloat16)
    lin = gemm.BlockwiseFP8Linear.from_linear(ref)
    x = torch.randn(16, 512, dtype=torch.bfloat16, device="xpu")
    out_fp = lin(x).float()
    out_ref = ref(x).float()
    rel = (out_fp - out_ref).abs().max().item()
    assert rel < 0.2


@xpu
def test_linearxpu_weight_memory_halved():
    torch.manual_seed(0)
    ref = torch.nn.Linear(5120, 5120, bias=False, device="xpu", dtype=torch.bfloat16)
    lin = gemm.BlockwiseFP8Linear.from_linear(ref)
    fp8_bytes = lin.weight.numel() * 1 + lin.weight_scale.numel() * 4
    bf16_bytes = ref.weight.numel() * 2
    assert fp8_bytes < bf16_bytes
    assert fp8_bytes == pytest.approx(bf16_bytes / 2, rel=0.01)
