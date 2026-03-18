from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemas.events import FrameEvent, WorkerMetadata
from schemas.world_state import TaskSnapshot, WorldStateSnapshot
from server.decision_engine import DecisionEngine


def test_decision_engine_requests_memory_only_for_active_memory_instruction() -> None:
    engine = DecisionEngine()
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
