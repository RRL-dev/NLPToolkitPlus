"""Cuda utils for Deep learning."""

from __future__ import annotations

from torch._C import device
from torch.cuda import is_available


def set_device() -> device:
    """Set device.

    Returns
    -------
        device: device of torch.

    """
    if is_available():
        return device(device=f"cuda:{0}")
    return device(device="cpu")
