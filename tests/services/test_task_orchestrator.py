from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control.behavior_fsm import BehaviorState
from ipc.messages import ActionStatus, TaskRequest
from memory.models import ObsObject
from perception.speaker_events import SpeakerEvent
from services.memory_service import MemoryService
from services.task_orchestrator import TaskOrchestrator


def test_task_orchestrator_attend_and_follow_state_transitions() -> None:
    memory_service = MemoryService()
    orchestrator = TaskOrchestrator(memory_service)

    orchestrator.on_speaker_event(SpeakerEvent(timestamp=1.0, direction_yaw_rad=0.6, speaker_id="caller"))
    command = orchestrator.step(now=1.1, robot_pose=(0.0, 0.0, 0.0))

    assert orchestrator.state == BehaviorState.ATTEND_CALLER
    assert command is not None
    assert command.action_type == "LOOK_AT"

    orchestrator.on_observations(
        [
            ObsObject(
                class_name="person",
                track_id="person_k",
                pose=(1.0, 0.0, 0.0),
                timestamp=2.0,
                confidence=0.9,
            )
        ]
    )
    orchestrator.submit_task(TaskRequest(command_text="따라와", speaker_id="caller"))
    follow_command = orchestrator.step(now=2.1, robot_pose=(0.0, 0.0, 0.0))

    assert orchestrator.state == BehaviorState.FOLLOW_TARGET
    assert follow_command is not None
    assert follow_command.action_type == "FOLLOW_TARGET"
    assert follow_command.target_track_id == "person_k"
    assert follow_command.target_person_id != ""

    recovery_command = orchestrator.step(
        now=2.2,
        robot_pose=(0.0, 0.0, 0.0),
        action_status=ActionStatus(command_id=follow_command.command_id, state="failed", reason="lost_target"),
    )

    assert orchestrator.state == BehaviorState.RECOVER_LOST_TARGET
    assert recovery_command is not None
    assert recovery_command.action_type == "NAV_TO_POSE"
    assert recovery_command.metadata["recovery"] in {"recent_exact_person_match", "spatial_reid_candidate", "last_visible_pose"}


def test_task_orchestrator_memory_recall_transitions_to_local_search() -> None:
    memory_service = MemoryService()
    memory_service.observe_objects(
        [
            ObsObject(
                class_name="apple",
                track_id="apple-track-1",
                pose=(3.0, 1.0, 0.0),
                timestamp=5.0,
                confidence=0.95,
                room_id="kitchen",
            )
        ]
    )
    memory_service.semantic_store.remember_rule("find:apple:kitchen", "prefer kitchen", succeeded=True)
    orchestrator = TaskOrchestrator(memory_service)
    request = TaskRequest(command_text="아까 봤던 사과를 찾아가", target_json={"target_class": "apple", "room_id": "kitchen"})
    orchestrator.submit_task(request)

    nav_command = orchestrator.step(now=5.1, robot_pose=(0.0, 0.0, 0.0))
    local_search_command = orchestrator.step(
        now=6.0,
        robot_pose=(2.8, 0.9, 0.0),
        action_status=ActionStatus(command_id=nav_command.command_id if nav_command else "", state="succeeded", success=True),
    )

    assert nav_command is not None
    assert nav_command.action_type == "NAV_TO_PLACE"
    assert orchestrator.state == BehaviorState.LOCAL_SEARCH
    assert local_search_command is not None
    assert local_search_command.action_type == "LOCAL_SEARCH"
