from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray | None = None
    centroid_xy: tuple[float, float] | None = None
    track_hint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorInfo:
    backend_name: str
    engine_path: str = ""
    warning: str = ""
    using_fallback: bool = False


class DetectorBackend(ABC):
    @property
    @abstractmethod
    def info(self) -> DetectorInfo:
        raise NotImplementedError

    @abstractmethod
    def detect(
        self,
        rgb_image: np.ndarray,
        *,
        timestamp: float,
        metadata: dict[str, Any] | None = None,
    ) -> list[DetectionResult]:
        raise NotImplementedError
