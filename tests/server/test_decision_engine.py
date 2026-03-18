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


def test_decision_engine_requests_memory_only_for_active_memory_instruction() -> None:
    engine = DecisionEngine(policy=RecoveryPolicy(4000, 1, 20000, 0, 1, 1000))
    frame_event = FrameEvent(
        metadata=WorkerMetadata(task_id="interactive", frame_id=4, timestamp_ns=time.time_ns(), source="test", timeout_ms=100),
        frame_id=4,
        timestamp_ns=time.time_ns(),
        source="test",
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        sim_time_s=1.0,
        observation=object(),
        batch=object(),
    )

    directive = engine.evaluate(
        world_state=WorldStateSnapshot(),
        task=TaskSnapshot(task_id="interactive", mode="interactive", state="active"),
        frame_event=frame_event,
        manual_command_present=True,
        active_memory_instruction="find the dock",
    )
    assert directive.process_perception is True
    assert directive.retrieve_memory is True
    assert directive.use_manual_command is True

    no_memory = engine.evaluate(
        world_state=WorldStateSnapshot(),
        task=TaskSnapshot(task_id="interactive", mode="interactive", state="idle"),
        frame_event=frame_event,
        manual_command_present=False,
        active_memory_instruction="",
    )
    assert no_memory.retrieve_memory is False
    assert no_memory.route_task_command is True
    assert no_memory.allow_planning is True


def test_decision_engine_suppresses_planning_during_recovery_backoff() -> None:
    engine = DecisionEngine(policy=RecoveryPolicy(4000, 1, 20000, 0, 1, 1000))
    now_ns = time.time_ns()
    frame_event = FrameEvent(
        metadata=WorkerMetadata(task_id="interactive", frame_id=9, timestamp_ns=now_ns, source="test", timeout_ms=100),
        frame_id=9,
        timestamp_ns=now_ns,
        source="test",
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        sim_time_s=1.0,
        observation=object(),
        batch=object(),
    )
    directive = engine.evaluate(
        world_state=WorldStateSnapshot(),
        task=TaskSnapshot(task_id="interactive", mode="interactive", state="active"),
        frame_event=frame_event,
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


def test_decision_engine_marks_fresh_planning_success_from_yaw_delta() -> None:
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
            planner_control_mode="yaw_delta",
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


def test_decision_engine_dual_policy_prefers_s1_before_refreshing_s2() -> None:
    engine = DecisionEngine()
    goal = type(
        "GoalCache",
        (),
        {"mode": "pixel_goal", "pixel_x": 11, "pixel_y": 22, "stop": False, "version": 3, "updated_at": 100.0},
    )()
    directive = engine.evaluate_dual(
        now=100.2,
        goal_cache=goal,
        traj_cache=None,
        last_s1_ts=0.0,
        last_s2_ts=100.0,
        s1_period_sec=0.2,
        s2_period_sec=1.0,
        goal_ttl_sec=3.0,
        traj_ttl_sec=1.5,
        traj_max_stale_sec=4.0,
        s2_retry_after_ts=0.0,
        force_s2_pending=False,
        events={},
    )
    assert directive.launch_s1 is True
    assert directive.launch_s2 is False
    assert directive.traj_missing is True
