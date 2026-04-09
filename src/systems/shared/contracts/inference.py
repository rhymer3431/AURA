from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class System2Result:
    """Normalized System2 response shared across inference and control."""

    status: str
    uv_norm: np.ndarray | None
    text: str
    latency_ms: float
    stamp_s: float
    pixel_xy: np.ndarray | None = None
    decision_mode: str | None = None
    action_sequence: tuple[str, ...] | None = None
    needs_requery: bool = False
    raw_payload: dict[str, Any] | None = None
