"""Minimal Depth Anything V2 encoder bundle used by NavDP."""

from .dinov2 import DINOv2 as _build_dinov2


def build_depth_anything_v2_encoder(model_name: str = "vits"):
    return _build_dinov2(model_name=model_name)


__all__ = ["build_depth_anything_v2_encoder"]
