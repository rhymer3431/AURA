from __future__ import annotations

import numpy as np

from .cv2_compat import cv2


def sanitize_depth_image(
    depth_m: np.ndarray,
    depth_max_m: float | None,
    *,
    clip_to_max: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth image must be [H,W] or [H,W,1], got {depth.shape}")
    valid_mask = np.isfinite(depth) & (depth > 0.0)
    fill_value = float(depth_max_m) if depth_max_m is not None else 0.0
    sanitized = np.nan_to_num(depth, nan=fill_value, posinf=fill_value, neginf=0.0)
    sanitized = np.maximum(sanitized, 0.0)
    if clip_to_max and depth_max_m is not None:
        sanitized = np.minimum(sanitized, float(depth_max_m))
    return sanitized, valid_mask


def compute_depth_display_range(
    depth_m: np.ndarray,
    *,
    default_max_m: float,
    min_percentile: float = 2.0,
    max_percentile: float = 98.0,
) -> tuple[float, float]:
    sanitized, valid_mask = sanitize_depth_image(depth_m, None, clip_to_max=False)
    if not np.any(valid_mask):
        return 0.0, float(default_max_m)

    valid_depth = sanitized[valid_mask]
    lower = float(np.percentile(valid_depth, float(min_percentile)))
    upper = float(np.percentile(valid_depth, float(max_percentile)))
    if not np.isfinite(lower):
        lower = float(np.min(valid_depth))
    if not np.isfinite(upper):
        upper = float(np.max(valid_depth))
    if upper <= lower + 1.0e-6:
        lower = float(np.min(valid_depth))
        upper = float(np.max(valid_depth))
    if upper <= lower + 1.0e-6:
        upper = lower + max(float(default_max_m), 1.0)
    return max(lower, 0.0), max(upper, lower + 1.0e-6)


def depth_to_heatmap_bgr(depth_m: np.ndarray, depth_max_m: float, *, depth_min_m: float = 0.0) -> np.ndarray:
    sanitized, valid_mask = sanitize_depth_image(depth_m, depth_max_m, clip_to_max=True)
    if sanitized.size == 0:
        return np.zeros((*sanitized.shape, 3), dtype=np.uint8)

    denom = max(float(depth_max_m) - float(depth_min_m), 1.0e-6)
    norm = 1.0 - np.clip((sanitized - float(depth_min_m)) / denom, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
    heatmap = np.stack((blue, green, red), axis=-1)
    heatmap = np.clip(np.round(heatmap * 255.0), 0.0, 255.0).astype(np.uint8)
    heatmap[~valid_mask] = 0
    return heatmap


def rgb_to_bgr(rgb_image: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb_image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"rgb image must be [H,W,3+], got {image.shape}")
    if image.shape[-1] == 4:
        rgba_to_bgr = getattr(cv2, "COLOR_RGBA2BGR", None)
        if rgba_to_bgr is not None:
            return cv2.cvtColor(image, rgba_to_bgr)
        image = image[..., :3]
    return cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR)


def build_rgb_depth_panel(
    rgb_image: np.ndarray,
    depth_m: np.ndarray,
    depth_max_m: float,
    *,
    depth_min_m: float = 0.0,
) -> np.ndarray:
    rgb_bgr = rgb_to_bgr(rgb_image)
    depth_bgr = depth_to_heatmap_bgr(depth_m, depth_max_m, depth_min_m=depth_min_m)
    if depth_bgr.shape[:2] != rgb_bgr.shape[:2]:
        depth_bgr = cv2.resize(depth_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.concatenate((rgb_bgr, depth_bgr), axis=1)
