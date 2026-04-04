from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


MemoryPolicyBackend = Literal["heuristic", "hf_generate"]


@dataclass(frozen=True)
class MemoryPolicyConfig:
    enabled: bool = False
    backend: MemoryPolicyBackend = "heuristic"
    shadow_mode: bool = True
    live_turns_enabled: bool = False
    live_stop_enabled: bool = False
    turn_yaw_delta_rad: float = math.pi / 6.0
    ambiguity_gap_threshold: float = 0.12
    model_name_or_path: str = ""
    device: str = ""
    max_new_tokens: int = 8
    temperature: float = 0.0
