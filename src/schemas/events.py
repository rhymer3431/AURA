from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from runtime.planning_session import ExecutionObservation


def _trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class WorkerMetadata:
    trace_id: str = field(default_factory=_trace_id)
    task_id: str = ""
    frame_id: int = -1
    timestamp_ns: int = field(default_factory=time.time_ns)
    source: str = ""
    timeout_ms: int = 0


@dataclass(frozen=True)
class FrameEvent:
    metadata: WorkerMetadata
    frame_id: int
    timestamp_ns: int
    source: str
    robot_pose_xyz: tuple[float, float, float]
    robot_yaw_rad: float
    sim_time_s: float
    observation: ExecutionObservation | None = None
    batch: IsaacObservationBatch | None = None
    sensor_meta: dict[str, Any] = field(default_factory=dict)
    planner_overlay: dict[str, Any] = field(default_factory=dict)
    publish_observation: bool = False
