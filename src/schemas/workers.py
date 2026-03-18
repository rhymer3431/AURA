from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

import numpy as np

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from inference.vlm import System2Request
from ipc.messages import ActionCommand
from memory.models import MemoryContextBundle
from runtime.planning_session import ExecutionObservation, TrajectoryUpdate
from schemas.commands import LocomotionProposal
from schemas.events import FrameEvent, WorkerMetadata

WorkerResultStatus = str


@dataclass(frozen=True)
class WorkerValidation:
    accepted: bool
    reason: str = ""


@dataclass(frozen=True, kw_only=True)
class WorkerRequestBase:
    metadata: WorkerMetadata = field(default_factory=WorkerMetadata)


@dataclass(frozen=True, kw_only=True)
class WorkerResultBase:
    metadata: WorkerMetadata = field(default_factory=WorkerMetadata)
    status: WorkerResultStatus = "ok"
    error: str = ""
    discard_reason: str = ""

    @property
    def ok(self) -> bool:
        return str(self.status) == "ok"


@dataclass(frozen=True, kw_only=True)
class PerceptionRequest(WorkerRequestBase):
    batch: IsaacObservationBatch | None = None
    publish: bool = True


@dataclass(frozen=True, kw_only=True)
class PerceptionResult(WorkerResultBase):
    batch: IsaacObservationBatch | None = None
    summary: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MemoryRequest(WorkerRequestBase):
    instruction: str = ""
    current_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, kw_only=True)
class MemoryResult(WorkerResultBase):
    memory_context: MemoryContextBundle | None = None
    summary: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class NavRequest(WorkerRequestBase):
    observation: ExecutionObservation | None = None
    action_command: ActionCommand | None = None
    robot_pos_world: np.ndarray | None = None
    robot_yaw: float = 0.0
    robot_quat_wxyz: np.ndarray | None = None
    image_bgr: np.ndarray | None = None
    depth_m: np.ndarray | None = None
    pixel_goal: tuple[int, int] | None = None
    sensor_meta: dict[str, Any] = field(default_factory=dict)
    cam_pos: np.ndarray | None = None
    cam_quat_wxyz: np.ndarray | None = None


@dataclass(frozen=True, kw_only=True)
class NavResult(WorkerResultBase):
    trajectory_update: TrajectoryUpdate | None = None
    trajectory_world: np.ndarray | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True, kw_only=True)
class LocomotionRequest(WorkerRequestBase):
    frame_idx: int = -1
    observation: ExecutionObservation | None = None
    action_command: ActionCommand | None = None
    trajectory_update: TrajectoryUpdate | None = None
    robot_pos_world: np.ndarray | None = None
    robot_lin_vel_world: np.ndarray | None = None
    robot_ang_vel_world: np.ndarray | None = None
    robot_yaw: float = 0.0
    robot_quat_wxyz: np.ndarray | None = None


@dataclass(frozen=True, kw_only=True)
class LocomotionResult(WorkerResultBase):
    proposal: LocomotionProposal | None = None


@dataclass(frozen=True, kw_only=True)
class S2Request(WorkerRequestBase):
    request: System2Request | None = None
    events: dict[str, Any] = field(default_factory=dict)
    memory_context: MemoryContextBundle | None = None


@dataclass(frozen=True, kw_only=True)
class S2Result(WorkerResultBase):
    mode: str = "wait"
    pixel_x: int | None = None
    pixel_y: int | None = None
    stop: bool = False
    yaw_delta_rad: float | None = None
    reason: str = ""
    latency_ms: float = 0.0
    source: str = "none"
    raw_text: str = ""
    history_frame_ids: tuple[int, ...] = ()
    needs_requery: bool = False


def stamp_worker_metadata(
    *,
    frame_event: FrameEvent | None = None,
    source: str,
    task_id: str | None = None,
    frame_id: int | None = None,
    timeout_ms: int | None = None,
    timestamp_ns: int | None = None,
    plan_version: int | None = None,
    goal_version: int | None = None,
    traj_version: int | None = None,
) -> WorkerMetadata:
    base = None if frame_event is None else frame_event.metadata
    default_meta = WorkerMetadata()
    return WorkerMetadata(
        trace_id=str(default_meta.trace_id if base is None else base.trace_id),
        task_id=str(base.task_id if task_id is None and base is not None else task_id or ""),
        frame_id=int(base.frame_id if frame_id is None and base is not None else (-1 if frame_id is None else frame_id)),
        timestamp_ns=int(base.timestamp_ns)
        if timestamp_ns is None and base is not None
        else int(time.time_ns() if timestamp_ns is None else timestamp_ns),
        source=str(source),
        timeout_ms=int(base.timeout_ms) if timeout_ms is None and base is not None else int(timeout_ms or 0),
        plan_version=base.plan_version if plan_version is None and base is not None else plan_version,
        goal_version=base.goal_version if goal_version is None and base is not None else goal_version,
        traj_version=base.traj_version if traj_version is None and base is not None else traj_version,
    )


def inherit_worker_metadata(
    metadata: WorkerMetadata,
    *,
    source: str | None = None,
    timestamp_ns: int | None = None,
    plan_version: int | None = None,
    goal_version: int | None = None,
    traj_version: int | None = None,
) -> WorkerMetadata:
    return WorkerMetadata(
        trace_id=str(metadata.trace_id),
        task_id=str(metadata.task_id),
        frame_id=int(metadata.frame_id),
        timestamp_ns=int(metadata.timestamp_ns if timestamp_ns is None else timestamp_ns),
        source=str(metadata.source if source is None else source),
        timeout_ms=int(metadata.timeout_ms),
        plan_version=metadata.plan_version if plan_version is None else plan_version,
        goal_version=metadata.goal_version if goal_version is None else goal_version,
        traj_version=metadata.traj_version if traj_version is None else traj_version,
    )


def metadata_timed_out(metadata: WorkerMetadata, *, now_ns: int | None = None) -> bool:
    timeout_ms = int(metadata.timeout_ms)
    if timeout_ms <= 0:
        return False
    current_ns = time.time_ns() if now_ns is None else int(now_ns)
    elapsed_ns = current_ns - int(metadata.timestamp_ns)
    return elapsed_ns > timeout_ms * 1_000_000


def validate_worker_metadata(
    metadata: WorkerMetadata,
    *,
    expected: WorkerMetadata | None = None,
    task_id: str | None = None,
    frame_id: int | None = None,
    plan_version: int | None = None,
    goal_version: int | None = None,
    traj_version: int | None = None,
) -> WorkerValidation:
    expected_task = str(expected.task_id) if expected is not None and str(expected.task_id) != "" else None
    explicit_task = None if task_id is None or str(task_id) == "" else str(task_id)
    if explicit_task is None:
        explicit_task = expected_task
    if explicit_task is not None and str(metadata.task_id) != explicit_task:
        return WorkerValidation(False, f"task_mismatch expected={explicit_task} actual={metadata.task_id}")

    expected_frame = int(expected.frame_id) if expected is not None and int(expected.frame_id) >= 0 else None
    explicit_frame = expected_frame if frame_id is None else int(frame_id)
    if explicit_frame is not None and int(metadata.frame_id) != explicit_frame:
        return WorkerValidation(False, f"frame_mismatch expected={explicit_frame} actual={metadata.frame_id}")

    for label, expected_value, actual_value in (
        ("plan_version", plan_version if plan_version is not None else (None if expected is None else expected.plan_version), metadata.plan_version),
        ("goal_version", goal_version if goal_version is not None else (None if expected is None else expected.goal_version), metadata.goal_version),
        ("traj_version", traj_version if traj_version is not None else (None if expected is None else expected.traj_version), metadata.traj_version),
    ):
        if expected_value is None:
            continue
        if actual_value is None or int(actual_value) != int(expected_value):
            return WorkerValidation(False, f"{label}_mismatch expected={expected_value} actual={actual_value}")
    return WorkerValidation(True)


_WorkerResultT = TypeVar("_WorkerResultT", bound=WorkerResultBase)


def reject_worker_result(result: _WorkerResultT, reason: str) -> _WorkerResultT:
    return replace(result, status="discarded", discard_reason=str(reason))


def build_error_result(result_type: type[_WorkerResultT], *, metadata: WorkerMetadata, error: str, **kwargs: Any) -> _WorkerResultT:
    return result_type(metadata=metadata, status="error", error=str(error), **kwargs)


def build_timeout_result(result_type: type[_WorkerResultT], *, metadata: WorkerMetadata, error: str = "", **kwargs: Any) -> _WorkerResultT:
    message = str(error).strip() or "worker result timed out"
    return result_type(metadata=metadata, status="timeout", error=message, **kwargs)


def finalize_worker_result(
    result: _WorkerResultT,
    *,
    expected: WorkerMetadata | None = None,
    now_ns: int | None = None,
    task_id: str | None = None,
    frame_id: int | None = None,
    plan_version: int | None = None,
    goal_version: int | None = None,
    traj_version: int | None = None,
) -> _WorkerResultT:
    if not result.ok:
        return result
    if metadata_timed_out(result.metadata, now_ns=now_ns):
        return replace(
            result,
            status="timeout",
            error=str(result.error).strip() or "worker result timed out",
        )
    validation = validate_worker_metadata(
        result.metadata,
        expected=expected,
        task_id=task_id,
        frame_id=frame_id,
        plan_version=plan_version,
        goal_version=goal_version,
        traj_version=traj_version,
    )
    if not validation.accepted:
        return reject_worker_result(result, validation.reason)
    return result
