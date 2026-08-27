import torch

from . import ops


def available() -> bool:
    """Whether the compiled XPU extension is importable and XPU is usable."""
    if not torch.xpu.is_available():
        return False
    try:
        import xpu_ltx_kernels._C  # noqa: PLC0415, F401

        return True
    except ImportError:
        return False


__all__ = ["available", "ops"]
