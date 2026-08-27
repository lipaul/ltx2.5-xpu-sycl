"""VAE neighborhood attention for the DiffVAE decoder on XPU.

A plain-SYCL ``na3d`` that mirrors ``natten.na3d`` semantics
(``[B, T, H, W, NH, HD]`` layout, clamped/shift-inward windows, scale applied
inside) so ltx-core's ``attention_function`` swap surface can route to it on
XPU instead of the Triton/eager fallbacks.
"""

from __future__ import annotations

from types import ModuleType

import torch


def _ext() -> ModuleType:
    import xpu_ltx_kernels._C  # noqa: PLC0415

    return xpu_ltx_kernels._C


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int | tuple[int, int, int],
    scale: float = 1.0,
) -> torch.Tensor:
    """3D neighborhood attention over ``[B, T, H, W, NH, HD]`` bf16 tensors.

    Matches ``natten.na3d(q, k, v, kernel_size, scale)``: scale is applied to
    the attention logits inside the kernel. For the ltx-core DiffVAE path the
    caller pre-scales Q and passes ``scale=1.0``.
    """
    if isinstance(kernel_size, int):
        kt = kh = kw = kernel_size
    else:
        kt, kh, kw = kernel_size
    return _ext().na3d(q, k, v, kt, kh, kw, scale)
