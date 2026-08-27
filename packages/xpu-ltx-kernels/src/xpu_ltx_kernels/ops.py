"""Python surface for the fused element ops, mirroring ``ltx_kernels.blockwise``
signatures so ltx-core can dispatch to them on XPU (see
``ltx_core.quantization.blockwise._impl``).
"""

from __future__ import annotations

from types import ModuleType

import torch


def _ext() -> ModuleType:
    import xpu_ltx_kernels._C  # noqa: PLC0415

    return xpu_ltx_kernels._C


def fp6_pack_tensor(x: torch.Tensor) -> torch.Tensor:
    """Pack FP8/uint8 weights to FP6 by dropping the e_1/e_2 exponent bits.
    ``[m, n]`` -> ``[m, n*3/4]`` uint8 (25% memory saving).
    """
    return _ext().fp6_pack(x)


def fp6_unpack_tensor(x: torch.Tensor, original_n: int) -> torch.Tensor:
    """Unpack FP6 back to FP8-with-zeroed-e1/e2 bytes: ``[m, n_packed]`` -> ``[m, original_n]``."""
    return _ext().fp6_unpack(x, original_n)


def rms_norm_rope(
    x: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    weights: torch.Tensor | None = None,
    out_fp8: bool = False,
) -> torch.Tensor:
    """Fused RMS-norm + interleaved RoPE on ``[*, h]`` bf16."""
    out = _ext().rms_norm_rope(x, cos_freqs, sin_freqs, weights)
    return out.to(torch.float8_e4m3fn) if out_fp8 else out


def rms_norm_split_rope(
    x: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    weights: torch.Tensor | None = None,
    out_fp8: bool = False,
) -> torch.Tensor:
    """Fused RMS-norm + split RoPE on ``[b, s, h]`` bf16."""
    out = _ext().rms_norm_split_rope(x, sin_freqs, cos_freqs, weights)
    return out.to(torch.float8_e4m3fn) if out_fp8 else out
