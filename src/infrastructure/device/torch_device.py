import torch

from config.settings import DEVICE


def get_device() -> torch.device:
    """Return configured torch device."""
    return DEVICE
