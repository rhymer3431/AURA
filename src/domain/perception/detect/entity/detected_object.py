from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DetectedObject:
    track_id: Optional[int]
    bbox: Tuple[int, int, int, int]
    score: float
    class_id: int
    class_name: str
