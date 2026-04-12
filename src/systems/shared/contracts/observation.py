"""Shared observation and hot-path runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
from typing import Any

import numpy as np


def _copy_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxWxC RGB image, got shape {image.shape}.")
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _empty_rgb_history() -> np.ndarray:
    return np.zeros((0, 0, 0, 3), dtype=np.uint8)


def normalize_rgb_history(rgb_history: np.ndarray | None) -> np.ndarray:
    if rgb_history is None:
        return _empty_rgb_history()
    array = np.asarray(rgb_history)
    if array.size == 0:
        return _empty_rgb_history()
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"rgb_history must have shape [T,H,W,3], got {array.shape}.")
    normalized = [_copy_rgb(frame) for frame in array]
    return np.stack(normalized, axis=0) if normalized else _empty_rgb_history()


def encode_rgb_history_npz(rgb_history: np.ndarray | None) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, rgb_history=normalize_rgb_history(rgb_history))
    return buffer.getvalue()


def decode_rgb_history_npz(payload: bytes | bytearray | memoryview | None) -> np.ndarray:
    if payload is None:
        return _empty_rgb_history()
    with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as archive:
        if "rgb_history" not in archive:
            raise ValueError("history_npz is missing rgb_history.")
        return normalize_rgb_history(archive["rgb_history"])


@dataclass(slots=True)
class RawObservation:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    camera_pos_w: np.ndarray
    camera_rot_w: np.ndarray
    robot_state: Any
    stamp_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObservationFrame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    camera_pos_w: np.ndarray
    camera_rot_w: np.ndarray
    robot_state: Any
    stamp_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerInput:
    robot_state: Any
    goal_xy_body: np.ndarray | None
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    camera_pos_w: np.ndarray
    camera_rot_w: np.ndarray
    stamp_s: float


@dataclass(slots=True)
class HistoryView:
    consumer: str
    session_id: str | None
    rgb_history: np.ndarray


@dataclass(slots=True)
class NavigationSessionSpec:
    instruction: str
    language: str
    task_id: str | None = None
    session_id: str | None = None


@dataclass(slots=True)
class TrajectoryPlan:
    trajectory_world_xy: np.ndarray
    stamp_s: float
    status: str = "idle"
    instruction: str | None = None
    task_id: str | None = None
    last_error: str | None = None


@dataclass(slots=True)
class LocomotionCommand:
    command: np.ndarray
    state_label: str
    reason: str | None = None
