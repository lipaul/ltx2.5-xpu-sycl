"""Torch-based XPU kernels for the blockwise FP8 Q8 path.

The CUDA blockwise path uses ltx-kernels Triton kernels for adanorm / rms-fma
quantization and gated attention. On XPU those Triton kernels are replaced by
these torch equivalents (same math, correct under torch XPU), while the
hot fused ops (``rms_norm_rope`` / ``rms_norm_split_rope``) come from
``xpu_ltx_kernels.ops``.
"""

from __future__ import annotations

import torch

BLOCK = 128
FP8_SCALE_MAX = 448.0


def blockwise_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise-quantize the last dim of ``x [b, s, h]`` to ``(fp8, scales)``."""
    b, s, h = x.shape
    xb = x.float().view(-1, h // BLOCK, BLOCK)
    absmax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = FP8_SCALE_MAX / absmax
    fp8 = (
        (xb * scale)
        .clamp(torch.finfo(torch.float8_e4m3fn).min, torch.finfo(torch.float8_e4m3fn).max)
        .to(torch.float8_e4m3fn)
    )
    return fp8.view(b, s, h), (1.0 / scale).view(b * s, h // BLOCK).contiguous().float()


def blockwise_quantize_adanorm(
    x: torch.Tensor,
    w: torch.Tensor | None,  # noqa: ARG001
    norm_scale: torch.Tensor,
    norm_shift: torch.Tensor,
    out_dtype: torch.dtype,
    hd_scale: float | None,  # noqa: ARG001
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS-norm(x) then affine by ``norm_scale/norm_shift``, blockwise-quantize."""
    b, s, h = x.shape
    rms = (x.float() * x.float()).mean(dim=-1, keepdim=True).rsqrt()
    xn = x.float() * rms
    y = xn * norm_scale.float() + norm_shift.float()
    if out_dtype != torch.float8_e4m3fn:
        raise ValueError(f"unsupported out_dtype {out_dtype}")
    return blockwise_quantize(y.view(b, s, h))


def blockwise_quantize_rms_fma(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS-norm(x), then ``x * gate + y`` fused, blockwise-quantized to fp8."""
    b, s, h = x.shape
    rms = (x.float() * x.float()).mean(dim=-1, keepdim=True).rsqrt()
    xn = x.float() * rms
    fused = xn * gate.float() + y.float()
    return blockwise_quantize(fused.view(b, s, h))


def gated_attention(attn_out: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    """Per-head gated attention: ``attn_out * 2*sigmoid(gate_logits)``.

    Matches ``ltx_kernels`` ``run_gated_attention`` (quantize=False path): x is
    ``[b, t, h]`` with ``h = nh * dim_head`` and one gate per (token, head);
    gates use ``2*sigmoid`` so a zero-init gate is identity.
    """
    b, t, h = attn_out.shape
    nh = gate_logits.shape[-1]
    dh = h // nh
    if gate_logits.dim() == 2:
        gate_logits = gate_logits.view(b, t, nh)
    gates = 2.0 * torch.sigmoid(gate_logits.float()).to(attn_out.dtype)
    return (attn_out.view(b, t, nh, dh) * gates.unsqueeze(-1)).view(b, t, h)


def blockwise_dequantize(payload: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Dequantize an ``(fp8, scales)`` blockwise payload back to bf16.

    Respects ``scales`` strides exactly like the Triton kernel (the adanorm
    path emits ``(1, tma_aligned_mn)`` strided scales).
    """
    x_fp8, scales = payload
    b, s, h = x_fp8.shape
    num_rows = b * s
    num_blocks = h // BLOCK
    sr, sc = scales.stride(0), scales.stride(1)
    sf = scales.flatten()
    rows = torch.arange(num_rows, device=scales.device)
    blks = torch.arange(num_blocks, device=scales.device)
    idx = rows[:, None] * sr + blks[None, :] * sc
    scale_t = sf[idx]  # [num_rows, num_blocks]
    xf = x_fp8.float().view(num_rows, num_blocks, BLOCK)
    return (xf * scale_t[:, :, None]).view(b, s, h).to(torch.bfloat16)
