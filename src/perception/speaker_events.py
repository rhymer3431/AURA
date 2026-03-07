from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SpeakerEvent:
    timestamp: float
    direction_yaw_rad: float
    speaker_id: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
