from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from systems.transport.messages import ActionStatus, TaskRequest
from memory.models import ObsObject
from perception.speaker_events import SpeakerEvent
from services.memory_service import MemoryService
from services.task_orchestrator import TaskOrchestrator


def test_speaker_event_binds_to_best_matching_person() -> None:
    memory_service = MemoryService()
    orchestrator = TaskOrchestrator(memory_service)

    orchestrator.on_speaker_event(SpeakerEvent(timestamp=1.0, direction_yaw_rad=0.15, speaker_id="caller"))
    orchestrator.on_observations(
        [
            ObsObject(
                class_name="person",
                track_id="p1",
                pose=(1.0, 0.2, 0.0),
                timestamp=1.1,
                confidence=0.95,
                metadata={"bearing_yaw_rad": 0.12, "appearance_signature": "blue", "depth_m": 1.02},
            ),
            ObsObject(
                class_name="person",
                track_id="p2",
                pose=(1.0, 2.0, 0.0),
                timestamp=1.1,
                confidence=0.95,
                metadata={"bearing_yaw_rad": 1.2, "appearance_signature": "red", "depth_m": 2.24},
            ),
        ]
    )

    binding_event = memory_service.temporal_store.last_event(event_type="speaker_binding")
    assert orchestrator.attention_service.bound_person_id != ""
    assert binding_event is not None
    assert binding_event.person_id == orchestrator.attention_service.bound_person_id
    assert binding_event.track_id == "p1"


def test_follow_target_reacquires_with_reid_after_track_change() -> None:
    memory_service = MemoryService()
    orchestrator = TaskOrchestrator(memory_service)
    orchestrator.on_observations(
        [
            ObsObject(
                class_name="person",
                track_id="p1",
                pose=(1.0, 0.0, 0.0),
                timestamp=2.0,
                confidence=0.9,
                metadata={"appearance_signature": "blue", "bearing_yaw_rad": 0.0, "depth_m": 1.0},
            )
        ]
    )
    orchestrator.submit_task(TaskRequest(command_text="follow me"))
    first_command = orchestrator.step(now=2.1, robot_pose=(0.0, 0.0, 0.0))

    orchestrator.on_observations(
        [
            ObsObject(
                class_name="person",
                track_id="p9",
                pose=(1.2, 0.1, 0.0),
                timestamp=2.3,
                confidence=0.9,
                metadata={"appearance_signature": "blue", "bearing_yaw_rad": 0.05, "depth_m": 1.2042},
            )
        ]
    )
    second_command = orchestrator.step(now=2.31, robot_pose=(0.0, 0.0, 0.0))
    recovery_command = orchestrator.step(
        now=2.4,
        robot_pose=(0.0, 0.0, 0.0),
        action_status=ActionStatus(command_id=second_command.command_id if second_command is not None else "", state="failed", reason="lost"),
    )

    assert first_command is not None
    assert second_command is not None
    assert first_command.target_person_id == second_command.target_person_id
    assert second_command.target_track_id == "p9"
    assert recovery_command is not None
    assert recovery_command.metadata["person_id"] == second_command.target_person_id
