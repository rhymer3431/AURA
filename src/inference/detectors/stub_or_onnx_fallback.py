from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import DetectionResult, DetectorBackend, DetectorInfo


@dataclass(frozen=True)
class ColorSegFallbackConfig:
    default_label: str = "object"
    color_name: str = "red"


class ColorSegFallbackDetector(DetectorBackend):
    def __init__(self, config: ColorSegFallbackConfig | None = None, *, warning: str = "") -> None:
        self._config = config or ColorSegFallbackConfig()
        self._info = DetectorInfo(
            backend_name="color_seg_fallback",
            warning=str(warning),
            using_fallback=True,
        )

    @property
    def info(self) -> DetectorInfo:
        return self._info

    def detect(
        self,
        rgb_image: np.ndarray,
        *,
        timestamp: float,
        metadata: dict[str, Any] | None = None,
    ) -> list[DetectionResult]:
        _ = timestamp
        metadata = dict(metadata or {})
        image = np.asarray(rgb_image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"rgb_image must be [H,W,3], got {image.shape}")

        label = str(metadata.get("target_class_hint", self._config.default_label)).strip() or self._config.default_label
        color_name = str(metadata.get("color_hint", self._config.color_name)).strip().lower() or self._config.color_name
        mask = self._build_mask(image, color_name=color_name)
        if not np.any(mask):
            return []

        ys, xs = np.nonzero(mask)
        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())
        centroid = (float(xs.mean()), float(ys.mean()))
        return [
            DetectionResult(
                class_name=label,
                confidence=0.95,
                bbox_xyxy=(x0, y0, x1, y1),
                mask=mask,
                centroid_xy=centroid,
                metadata={"color_hint": color_name},
            )
        ]

    @staticmethod
    def _build_mask(image: np.ndarray, *, color_name: str) -> np.ndarray:
        red = image[..., 0].astype(np.int16)
        green = image[..., 1].astype(np.int16)
        blue = image[..., 2].astype(np.int16)
        if color_name == "blue":
            mask = (blue > 150) & (blue - red > 40) & (blue - green > 40)
        elif color_name == "green":
            mask = (green > 150) & (green - red > 40) & (green - blue > 40)
        else:
            mask = (red > 150) & (red - green > 40) & (red - blue > 40)
        return mask.astype(bool)
