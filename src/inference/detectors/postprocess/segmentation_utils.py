from __future__ import annotations

from typing import Any

import numpy as np


def clamp_bbox_xyxy(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_xyxy
    x0_i = int(np.clip(np.round(float(x0)), 0, max(int(width) - 1, 0)))
    y0_i = int(np.clip(np.round(float(y0)), 0, max(int(height) - 1, 0)))
    x1_i = int(np.clip(np.round(float(x1)), x0_i, max(int(width) - 1, x0_i)))
    y1_i = int(np.clip(np.round(float(y1)), y0_i, max(int(height) - 1, y0_i)))
    return x0_i, y0_i, x1_i, y1_i


def resize_mask_nearest(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    height, width = int(output_shape[0]), int(output_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid output shape: {output_shape!r}")
    src = np.asarray(mask, dtype=np.float32)
    ys = np.clip(np.round(np.linspace(0, max(src.shape[0] - 1, 0), height)).astype(np.int32), 0, max(src.shape[0] - 1, 0))
    xs = np.clip(np.round(np.linspace(0, max(src.shape[1] - 1, 0), width)).astype(np.int32), 0, max(src.shape[1] - 1, 0))
    return src[np.ix_(ys, xs)]


def mask_from_proto_coeff(
    proto: np.ndarray,
    coeff: np.ndarray,
    *,
    output_shape: tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    proto_arr = np.asarray(proto, dtype=np.float32)
    coeff_arr = np.asarray(coeff, dtype=np.float32).reshape(-1)
    if proto_arr.ndim != 3:
        raise ValueError(f"proto must be [C,H,W], got {proto_arr.shape}")
    if proto_arr.shape[0] != coeff_arr.shape[0]:
        raise ValueError(f"mask coeff dim mismatch: proto={proto_arr.shape} coeff={coeff_arr.shape}")
    logits = np.tensordot(coeff_arr, proto_arr, axes=(0, 0))
    probs = 1.0 / (1.0 + np.exp(-logits))
    resized = resize_mask_nearest(probs, output_shape)
    return np.asarray(resized >= float(threshold), dtype=bool)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    mask_arr = np.asarray(mask, dtype=bool)
    if not np.any(mask_arr):
        return None
    ys, xs = np.nonzero(mask_arr)
    return float(xs.mean()), float(ys.mean())


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    mask_arr = np.asarray(mask, dtype=bool)
    if not np.any(mask_arr):
        return None
    ys, xs = np.nonzero(mask_arr)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def decode_mask_metadata(mask: np.ndarray) -> dict[str, Any]:
    centroid = mask_centroid(mask)
    bbox = bbox_from_mask(mask)
    return {
        "mask_area_px": int(np.asarray(mask, dtype=bool).sum()),
        "mask_centroid_xy": None if centroid is None else [float(centroid[0]), float(centroid[1])],
        "mask_bbox_xyxy": None if bbox is None else list(bbox),
    }
