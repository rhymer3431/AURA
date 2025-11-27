from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DetectedObject:
    track_id: Optional[int]
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    class_id: int
    class_name: str

    @property
    def label(self) -> str:  # backward compat
        return self.class_name

    def as_bbox(self):
        return list(self.bbox)

    def __repr__(self):
        return f"DetectedObject(label={self.class_name}, score={self.score:.2f})"
