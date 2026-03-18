from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemas.events import FrameEvent, WorkerMetadata
from schemas.recovery import RecoveryState, RecoveryStateSnapshot
from server.safety_supervisor import SafetySupervisor


class _Args:
    safety_timeout_sec = 0.02
    traj_max_stale_sec = 4.0
    retry = 1
    sensor_wait_budget_ms = 50
    recovery_turn_retry_limit = 1
    s2_retry_backoff_ms = 100
    s2_period_sec = 1.0


def _frame(*, observation: object | None = object(), batch: object | None = object(), metadata_ts: int | None = None) -> FrameEvent:
    ts = time.time_ns()
    return FrameEvent(
        metadata=WorkerMetadata(task_id="interactive", frame_id=4, timestamp_ns=ts if metadata_ts is None else metadata_ts, source="test", timeout_ms=100),
        frame_id=4,
        timestamp_ns=ts,
        source="test",
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        sim_time_s=1.0,
        observation=observation,
        batch=batch,
    )


def test_safety_supervisor_waits_for_sensor_then_safe_stops() -> None:
    supervisor = SafetySupervisor(_Args())
    waiting = supervisor.evaluate(
        frame_event=_frame(observation=None, batch=None),
        trajectory_update=SimpleNamespace(stale_sec=0.0),
        recovery_state=RecoveryStateSnapshot.normal(),
        now_ns=10,
    )
    assert waiting.recovery_state.state == RecoveryState.WAIT_SENSOR

    safe_stop = supervisor.evaluate(
        frame_event=_frame(observation=None, batch=None),
        trajectory_update=SimpleNamespace(stale_sec=0.0),
        recovery_state=waiting.recovery_state,
        now_ns=100_000_000,
    )
    assert safe_stop.recovery_state.state == RecoveryState.SAFE_STOP


def test_safety_supervisor_times_out_to_safe_stop() -> None:
    supervisor = SafetySupervisor(_Args())
    stale_frame = _frame(metadata_ts=time.time_ns() + 100_000_000)
    decision = supervisor.evaluate(
        frame_event=stale_frame,
        trajectory_update=SimpleNamespace(stale_sec=0.0),
        recovery_state=RecoveryStateSnapshot.normal(),
        now_ns=time.time_ns(),
    )
    assert decision.recovery_state.state == RecoveryState.SAFE_STOP
    assert decision.safety_reason == "timeout"


def test_safety_supervisor_marks_stale_trajectory_for_replan() -> None:
    supervisor = SafetySupervisor(_Args())
    decision = supervisor.evaluate(
        frame_event=_frame(),
        trajectory_update=SimpleNamespace(stale_sec=6.0),
        recovery_state=RecoveryStateSnapshot.normal(),
        now_ns=time.time_ns(),
    )
    assert decision.recovery_state.state == RecoveryState.REPLAN_PENDING
    assert decision.safety_reason == "trajectory_stale"
