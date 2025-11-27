# src/robotics/domain/detection/detected_object.py

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DetectedObject:
    track_id: Optional[int]  # 추적 ID
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float  # detection confidence
    class_id: int
    class_name: str
