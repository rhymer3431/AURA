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
