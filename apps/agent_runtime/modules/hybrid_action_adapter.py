from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


_MAP_PATH = Path(__file__).resolve().parent / "g1_joint_map.json"
MAX_JOINT_DELTA_PER_STEP = 0.05
TRANSITION_STEPS = 10


def _load_joint_map(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Joint map file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_group_indices(path: Path) -> Dict[str, List[int]]:
    data = _load_joint_map(path)
    groups: Dict[str, List[int]] = {}
    joints = sorted(data.get("joints", []), key=lambda j: int(j["sonic_idx"]))
    for joint in joints:
        idx = int(joint["sonic_idx"])
        group = str(joint["group"])
        groups.setdefault(group, []).append(idx)
    return groups


GROUP_INDICES = _build_group_indices(_MAP_PATH)
LEFT_LEG = GROUP_INDICES.get("left_leg", [])
RIGHT_LEG = GROUP_INDICES.get("right_leg", [])
WAIST = GROUP_INDICES.get("waist", [])
LEFT_ARM = GROUP_INDICES.get("left_arm", [])
RIGHT_ARM = GROUP_INDICES.get("right_arm", [])


def _as_vec(x, dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros((dim,), dtype=np.float32)
    n = min(dim, arr.shape[0])
    out[:n] = arr[:n]
    return out


class HybridActionAdapter:
    def __init__(
        self,
        max_joint_delta_per_step: float = MAX_JOINT_DELTA_PER_STEP,
        transition_steps: int = TRANSITION_STEPS,
    ) -> None:
        self.max_joint_delta_per_step = max(0.0, float(max_joint_delta_per_step))
        self.transition_steps = max(0, int(transition_steps))
        self._prev_merged: Optional[np.ndarray] = None
        self._transition_step = 0
        self._prev_mode: Optional[str] = None

    def reset(self) -> None:
        self._prev_merged = None
        self._transition_step = 0
        self._prev_mode = None

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        return str(mode).strip().lower()

    def _compute_merge(
        self,
        groot_actions: Dict,
        sonic_vec: np.ndarray,
        mode_key: str,
        waist_blend_alpha: float,
    ) -> np.ndarray:
        merged = sonic_vec.copy()

        if mode_key == "locomotion":
            return merged

        if mode_key == "idle":
            for idx in LEFT_ARM + RIGHT_ARM:
                merged[idx] = 0.0
            return merged

        if mode_key != "manipulation":
            return merged

        left_arm_cmd = _as_vec(groot_actions.get("left_arm", []), len(LEFT_ARM))
        right_arm_cmd = _as_vec(groot_actions.get("right_arm", []), len(RIGHT_ARM))
        waist_cmd = _as_vec(groot_actions.get("waist", []), len(WAIST))

        for i, idx in enumerate(LEFT_ARM):
            merged[idx] = left_arm_cmd[i]
        for i, idx in enumerate(RIGHT_ARM):
            merged[idx] = right_arm_cmd[i]
        for i, idx in enumerate(WAIST):
            merged[idx] = waist_blend_alpha * waist_cmd[i] + (1.0 - waist_blend_alpha) * sonic_vec[idx]
        return merged

    def merge(
        self,
        groot_actions: Dict,
        sonic_actions: np.ndarray,
        mode: str,
        waist_blend_alpha: float = 0.3,
    ) -> np.ndarray:
        sonic_vec = np.asarray(sonic_actions, dtype=np.float32)
        if sonic_vec.ndim == 2:
            sonic_vec = sonic_vec[0]
        sonic_vec = _as_vec(sonic_vec, 29)

        mode_key = self._normalize_mode(mode)
        if self._prev_mode is not None and mode_key != self._prev_mode:
            self._transition_step = self.transition_steps
        self._prev_mode = mode_key

        merged_target = self._compute_merge(groot_actions, sonic_vec, mode_key, waist_blend_alpha)

        if self._transition_step > 0 and self._prev_merged is not None and self.transition_steps > 0:
            alpha = 1.0 - (self._transition_step / float(self.transition_steps))
            merged_target = ((1.0 - alpha) * self._prev_merged + alpha * merged_target).astype(np.float32)
            self._transition_step -= 1

        if self._prev_merged is not None and self.max_joint_delta_per_step > 0.0:
            delta = merged_target - self._prev_merged
            delta = np.clip(delta, -self.max_joint_delta_per_step, self.max_joint_delta_per_step)
            merged_target = (self._prev_merged + delta).astype(np.float32)

        self._prev_merged = merged_target.copy()
        return merged_target


_DEFAULT_ADAPTER = HybridActionAdapter()


def reset() -> None:
    _DEFAULT_ADAPTER.reset()


def merge(
    groot_actions: Dict,
    sonic_actions: np.ndarray,
    mode: str,
    waist_blend_alpha: float = 0.3,
) -> np.ndarray:
    """
    Merge GR00T upper-body action with SONIC whole-body action.

    Args:
        groot_actions: GR00T action dict (left_arm/right_arm/waist keys).
        sonic_actions: SONIC action vector, shape (29,) or (1, 29).
        mode: "manipulation" | "locomotion" | "idle"
        waist_blend_alpha: GR00T weight for waist in manipulation mode.
    """
    return _DEFAULT_ADAPTER.merge(
        groot_actions=groot_actions,
        sonic_actions=sonic_actions,
        mode=mode,
        waist_blend_alpha=waist_blend_alpha,
    )
