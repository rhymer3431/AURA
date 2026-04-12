"""Encoding helpers for navigation-system observation exchange."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def encode_bytes_base64(payload: bytes) -> str:
    return base64.b64encode(bytes(payload)).decode("ascii")


def decode_bytes_base64(payload: str) -> bytes:
    return base64.b64decode(str(payload).encode("ascii"))


def encode_rgb_jpeg_base64(rgb: np.ndarray) -> str:
    image = np.asarray(rgb)
    if image.ndim != 3:
        raise ValueError(f"Expected HxWxC RGB image, got shape {image.shape}.")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=95)
    return encode_bytes_base64(buffer.getvalue())


def encode_depth_png_base64(depth: np.ndarray) -> str:
    image = np.asarray(depth, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"Expected HxW depth image, got shape {image.shape}.")
    image_mm = np.rint(np.clip(image, 0.0, 6.5535) * 10000.0).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(image_mm).save(buffer, format="PNG")
    return encode_bytes_base64(buffer.getvalue())


def decode_rgb_jpeg_base64(payload: str) -> np.ndarray:
    buffer = io.BytesIO(decode_bytes_base64(payload))
    image = Image.open(buffer).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def decode_depth_png_base64(payload: str) -> np.ndarray:
    buffer = io.BytesIO(decode_bytes_base64(payload))
    image = Image.open(buffer)
    depth_mm = np.asarray(image, dtype=np.uint16)
    return depth_mm.astype(np.float32) / 10000.0
