from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.messages import ActionStatus, TaskRequest
from memory.models import ObsObject
from services.memory_service import MemoryService
from services.task_orchestrator import TaskOrchestrator


def test_object_recall_replans_to_next_candidate_after_failure() -> None:
    memory_service = MemoryService()
    memory_service.observe_objects(
        [
            ObsObject(
                class_name="apple",
                track_id="apple-track-1",
                pose=(2.0, 0.0, 0.0),
                timestamp=10.0,
                confidence=0.80,
                room_id="kitchen",
            ),
            ObsObject(
                class_name="apple",
                track_id="apple-track-2",
                pose=(4.0, 0.0, 0.0),
                timestamp=11.0,
                confidence=0.95,
                room_id="kitchen",
            ),
        ]
    )
    memory_service.semantic_store.remember_rule("find:apple:kitchen", "prefer kitchen", succeeded=True)
    orchestrator = TaskOrchestrator(memory_service)
    request = TaskRequest(command_text="아까 봤던 사과를 찾아가", target_json={"target_class": "apple", "room_id": "kitchen"})
    orchestrator.submit_task(request)

    first_command = orchestrator.step(now=11.1, robot_pose=(0.0, 0.0, 0.0))
    second_command = orchestrator.step(
        now=11.2,
        robot_pose=(0.0, 0.0, 0.0),
        action_status=ActionStatus(
            command_id=first_command.command_id if first_command is not None else "",
            state="failed",
            reason="blocked",
            success=False,
        ),
    )

    assert first_command is not None
    assert second_command is not None
    assert first_command.action_type == "NAV_TO_PLACE"
    assert second_command.action_type == "NAV_TO_PLACE"
    assert first_command.metadata["candidate_id"] != second_command.metadata["candidate_id"]
    assert "semantic_rule_keys" in first_command.metadata
