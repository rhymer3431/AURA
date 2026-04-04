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
from schemas.world_state import TaskSnapshot, WorldStateSnapshot
from server.decision_engine import DecisionEngine
from server.recovery_state import RecoveryPolicy


def _frame_event(now_ns: int | None = None) -> FrameEvent:
    timestamp_ns = time.time_ns() if now_ns is None else int(now_ns)
    return FrameEvent(
        metadata=WorkerMetadata(task_id="interactive", frame_id=4, timestamp_ns=timestamp_ns, source="test", timeout_ms=100),
        frame_id=4,
        timestamp_ns=timestamp_ns,
        source="test",
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        sim_time_s=1.0,
        observation=object(),
        batch=object(),
    )


def test_decision_engine_allows_planning_for_active_navigation_frame() -> None:
    engine = DecisionEngine(policy=RecoveryPolicy(4000, 1, 20000, 0, 1, 1000))

    directive = engine.evaluate(
        world_state=WorldStateSnapshot(),
        task=TaskSnapshot(task_id="interactive", mode="interactive", state="active"),
        frame_event=_frame_event(),
        manual_command_present=True,
        active_memory_instruction="find the dock",
    )

    assert directive.process_perception is True
    assert directive.retrieve_memory is False
    assert directive.use_manual_command is True
    assert directive.allow_planning is True
    assert directive.reason == "mode:NAV"


def test_decision_engine_suppresses_planning_during_recovery_backoff() -> None:
    engine = DecisionEngine(policy=RecoveryPolicy(4000, 1, 20000, 0, 1, 1000))
    now_ns = time.time_ns()

    directive = engine.evaluate(
        world_state=WorldStateSnapshot(),
        task=TaskSnapshot(task_id="interactive", mode="interactive", state="active"),
        frame_event=_frame_event(now_ns),
        manual_command_present=False,
        active_memory_instruction="dock",
        recovery_state=RecoveryStateSnapshot(
            current_state=RecoveryState.REPLAN_PENDING.value,
            entered_at_ns=now_ns - 100,
            backoff_until_ns=now_ns + 1_000_000,
            last_trigger_reason="nav_failed",
        ),
        now_ns=now_ns,
    )
    assert directive.allow_planning is False
    assert directive.backoff_active is True
    assert directive.reason == "recovery_backoff"


def test_decision_engine_marks_fresh_planning_success_from_wait_or_stop_modes() -> None:
    engine = DecisionEngine(policy=RecoveryPolicy(4000, 1, 20000, 0, 1, 1000))

    evaluation = engine.evaluate_planning_outcome(
        world_state=WorldStateSnapshot(),
        task=TaskSnapshot(task_id="interactive", mode="interactive", state="active"),
        recovery_state=RecoveryStateSnapshot(current_state=RecoveryState.RECOVERY_TURN.value, entered_at_ns=1),
        trajectory_update=SimpleNamespace(
            stats=SimpleNamespace(last_error=""),
            plan_version=0,
            goal_version=0,
            traj_version=0,
            planner_control_mode="wait",
            stale_sec=0.0,
        ),
        action_command=SimpleNamespace(action_type="LOCAL_SEARCH"),
        now_ns=time.time_ns(),
    )
    assert evaluation.recovery_state.state == RecoveryState.NORMAL


def test_decision_engine_promotes_repeated_planner_failures() -> None:
    engine = DecisionEngine(policy=RecoveryPolicy(4000, 1, 20000, 0, 1, 1000))
    now_ns = time.time_ns()
    first = engine.evaluate_planning_outcome(
        world_state=WorldStateSnapshot(),
        task=TaskSnapshot(task_id="interactive", mode="interactive", state="active"),
        recovery_state=RecoveryStateSnapshot.normal(),
        trajectory_update=SimpleNamespace(
            stats=SimpleNamespace(last_error="planner failed"),
            plan_version=0,
            goal_version=0,
            traj_version=0,
            planner_control_mode=None,
        ),
        action_command=SimpleNamespace(action_type="LOCAL_SEARCH"),
        now_ns=now_ns,
    )
    second = engine.evaluate_planning_outcome(
        world_state=WorldStateSnapshot(),
        task=TaskSnapshot(task_id="interactive", mode="interactive", state="active"),
        recovery_state=first.recovery_state,
        trajectory_update=SimpleNamespace(
            stats=SimpleNamespace(last_error="planner failed"),
            plan_version=0,
            goal_version=0,
            traj_version=0,
            planner_control_mode=None,
        ),
        action_command=SimpleNamespace(action_type="LOCAL_SEARCH"),
        now_ns=now_ns + 1,
    )
    assert first.recovery_state.state == RecoveryState.REPLAN_PENDING
    assert second.recovery_state.state == RecoveryState.RECOVERY_TURN
