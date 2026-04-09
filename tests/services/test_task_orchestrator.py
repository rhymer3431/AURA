from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aura_config.memory_policy_config import MemoryPolicyConfig
from control.behavior_fsm import BehaviorState
from systems.transport.messages import ActionStatus, TaskRequest
from memory.models import ObsObject
from perception.speaker_events import SpeakerEvent
from services.memory_policy_types import MemoryPolicyContext, MemoryPolicyDecision, MemoryPolicyLabel
from services.memory_service import MemoryService
from services.task_orchestrator import TaskOrchestrator, TaskOrchestratorConfig


def _record_perception_frame(
    memory_service: MemoryService,
    *,
    frame_id: int,
    class_name: str,
    track_id: str,
    pose: tuple[float, float, float],
    room_id: str,
    timestamp: float,
    confidence: float = 0.95,
    bearing_yaw_rad: float = 0.0,
) -> None:
    memory_service.record_perception_frame(
        frame_id=frame_id,
        rgb_image=np.zeros((24, 24, 3), dtype=np.uint8),
        observations=[
            ObsObject(
                class_name=class_name,
                track_id=track_id,
                pose=pose,
                timestamp=timestamp,
                confidence=confidence,
                room_id=room_id,
                metadata={"depth_m": max(float(pose[0]), 0.5), "bearing_yaw_rad": bearing_yaw_rad},
            )
        ],
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
    )


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
                metadata={"depth_m": 1.0},
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


def test_task_orchestrator_memory_policy_turn_left_then_replans() -> None:
    memory_service = MemoryService()
    memory_service.set_planner_task(
        instruction="아까 봤던 사과를 찾아가",
        planner_mode="interactive",
        task_state="active",
    )
    _record_perception_frame(
        memory_service,
        frame_id=1,
        class_name="apple",
        track_id="apple-left",
        pose=(3.0, 0.0, 0.0),
        room_id="kitchen",
        timestamp=5.0,
        bearing_yaw_rad=-0.7,
    )
    orchestrator = TaskOrchestrator(
        memory_service,
        config=TaskOrchestratorConfig(
            memory_policy=MemoryPolicyConfig(enabled=True, shadow_mode=False, live_turns_enabled=True)
        ),
    )
    request = TaskRequest(command_text="아까 봤던 사과를 찾아가", target_json={"target_class": "apple", "room_id": "kitchen"})
    orchestrator.submit_task(request)

    turn_command = orchestrator.step(now=5.1, robot_pose=(0.0, 0.0, 0.0), robot_yaw_rad=0.0)
    nav_command = orchestrator.step(
        now=5.3,
        robot_pose=(0.0, 0.0, 0.0),
        robot_yaw_rad=float(np.pi / 6.0),
        action_status=ActionStatus(command_id=turn_command.command_id if turn_command else "", state="succeeded", success=True),
    )

    assert turn_command is not None
    assert turn_command.action_type == "LOOK_AT"
    assert turn_command.metadata["memory_policy"] == "TURN_LEFT"
    assert nav_command is not None
    assert nav_command.action_type == "NAV_TO_PLACE"


def test_task_orchestrator_memory_policy_turn_right_then_replans() -> None:
    memory_service = MemoryService()
    memory_service.set_planner_task(
        instruction="아까 지나쳤던 사과 쪽으로 다시 가줘",
        planner_mode="interactive",
        task_state="active",
    )
    _record_perception_frame(
        memory_service,
        frame_id=1,
        class_name="apple",
        track_id="apple-right",
        pose=(3.0, 0.0, 0.0),
        room_id="hallway",
        timestamp=5.0,
        bearing_yaw_rad=0.7,
    )
    orchestrator = TaskOrchestrator(
        memory_service,
        config=TaskOrchestratorConfig(
            memory_policy=MemoryPolicyConfig(enabled=True, shadow_mode=False, live_turns_enabled=True)
        ),
    )
    request = TaskRequest(command_text="아까 지나쳤던 사과 쪽으로 다시 가줘", target_json={"target_class": "apple", "room_id": "hallway"})
    orchestrator.submit_task(request)

    turn_command = orchestrator.step(now=5.1, robot_pose=(0.0, 0.0, 0.0), robot_yaw_rad=0.0)
    nav_command = orchestrator.step(
        now=5.3,
        robot_pose=(0.0, 0.0, 0.0),
        robot_yaw_rad=float(-np.pi / 6.0),
        action_status=ActionStatus(command_id=turn_command.command_id if turn_command else "", state="succeeded", success=True),
    )

    assert turn_command is not None
    assert turn_command.action_type == "LOOK_AT"
    assert turn_command.metadata["memory_policy"] == "TURN_RIGHT"
    assert nav_command is not None
    assert nav_command.action_type == "NAV_TO_PLACE"


def test_task_orchestrator_shadow_wait_falls_back_to_memory_recall(monkeypatch) -> None:  # noqa: ANN001
    memory_service = MemoryService()
    memory_service.observe_objects(
        [
            ObsObject(
                class_name="apple",
                track_id="apple-a",
                pose=(2.0, 0.0, 0.0),
                timestamp=10.0,
                confidence=0.9,
                room_id="kitchen",
            )
        ]
    )
    orchestrator = TaskOrchestrator(
        memory_service,
        config=TaskOrchestratorConfig(
            memory_policy=MemoryPolicyConfig(enabled=True, shadow_mode=False, live_turns_enabled=True)
        ),
    )
    request = TaskRequest(command_text="아까 봤던 사과를 찾아가", target_json={"target_class": "apple", "room_id": "kitchen"})
    orchestrator.submit_task(request)

    def _fake_evaluate_remembered_object(**kwargs):  # noqa: ANN003
        return (
            MemoryPolicyDecision(
                label=MemoryPolicyLabel.WAIT,
                confidence=0.6,
                source="heuristic",
                fallback_used=False,
                shadow_only=True,
                feature_snapshot={"ambiguity": True},
            ),
            "Return exactly one label.",
            MemoryPolicyContext(
                instruction=request.command_text,
                target_class="apple",
                task_state="active",
                current_pose=(0.0, 0.0, 0.0),
                visible_target_now=False,
                memory_context=None,
                recall_result=None,
            ),
        )

    monkeypatch.setattr(orchestrator.memory_policy_service, "evaluate_remembered_object", _fake_evaluate_remembered_object)

    nav_command = orchestrator.step(now=10.1, robot_pose=(0.0, 0.0, 0.0), robot_yaw_rad=0.0)

    snapshot = orchestrator.snapshot()
    assert nav_command is not None
    assert nav_command.action_type == "NAV_TO_PLACE"
    assert snapshot["last_memory_policy"]["label"] == "WAIT"
    assert snapshot["last_memory_policy"]["shadow_only"] is True


def test_task_orchestrator_shadow_direct_vision_falls_back_to_memory_recall(monkeypatch) -> None:  # noqa: ANN001
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
    orchestrator = TaskOrchestrator(
        memory_service,
        config=TaskOrchestratorConfig(memory_policy=MemoryPolicyConfig(enabled=True, shadow_mode=False)),
    )
    request = TaskRequest(command_text="아까 봤던 사과를 찾아가", target_json={"target_class": "apple", "room_id": "kitchen"})
    orchestrator.submit_task(request)

    def _fake_evaluate_remembered_object(**kwargs):  # noqa: ANN003
        return (
            MemoryPolicyDecision(
                label=MemoryPolicyLabel.ROUTE_DIRECT_VISION,
                confidence=0.8,
                source="heuristic",
                fallback_used=False,
                shadow_only=True,
                feature_snapshot={"visible_target_now": True},
            ),
            "Return exactly one label.",
            MemoryPolicyContext(
                instruction=request.command_text,
                target_class="apple",
                task_state="active",
                current_pose=(0.0, 0.0, 0.0),
                visible_target_now=True,
                memory_context=None,
                recall_result=None,
            ),
        )

    monkeypatch.setattr(orchestrator.memory_policy_service, "evaluate_remembered_object", _fake_evaluate_remembered_object)
    nav_command = orchestrator.step(now=5.1, robot_pose=(0.0, 0.0, 0.0), robot_yaw_rad=0.0)

    snapshot = orchestrator.snapshot()
    assert nav_command is not None
    assert nav_command.action_type == "NAV_TO_PLACE"
    assert snapshot["last_memory_policy"]["label"] == "ROUTE_DIRECT_VISION"
    assert snapshot["last_memory_policy"]["shadow_only"] is True


def test_task_orchestrator_visible_object_target_flows_to_nav_and_local_search_fallback() -> None:
    memory_service = MemoryService()
    orchestrator = TaskOrchestrator(memory_service)
    orchestrator.on_observations(
        [
            ObsObject(
                class_name="apple",
                track_id="apple_live",
                pose=(3.0, 0.0, 0.4),
                timestamp=1.0,
                confidence=0.95,
                metadata={"depth_m": 3.0},
            )
        ]
    )
    request = TaskRequest(
        command_text="보이는 사과로 가",
        target_json={"target_mode": "goto_visible_object", "target_class": "apple"},
    )
    orchestrator.submit_task(request)

    nav_command = orchestrator.step(now=1.1, robot_pose=(0.0, 0.0, 0.0))
    search_command = orchestrator.step(now=2.7, robot_pose=(0.0, 0.0, 0.0))
    stop_command = orchestrator.step(
        now=2.8,
        robot_pose=(0.0, 0.0, 0.0),
        action_status=ActionStatus(command_id=search_command.command_id if search_command else "", state="failed", reason="not_found"),
    )

    assert orchestrator.state == BehaviorState.IDLE
    assert nav_command is not None
    assert nav_command.action_type == "NAV_TO_POSE"
    assert nav_command.metadata["target_mode"] == "goto_visible_object"
    assert nav_command.metadata["pose_source"] == "filtered_track"
    assert nav_command.target_pose_xyz is not None
    assert round(float(nav_command.target_pose_xyz[0]), 4) == 2.1
    assert search_command is not None
    assert search_command.action_type == "LOCAL_SEARCH"
    assert search_command.metadata["recovery"] == "visible_target_local_search"
    assert stop_command is not None
    assert stop_command.action_type == "STOP"
