from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrackedObject:
    track_id: int
    bbox: Tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str
    velocity: Optional[Tuple[float, float]] = None
    timestamp: Optional[float] = None

    @property
    def label(self) -> str:  # compat with any old callers
        return self.class_name

    ...
