from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.messages import FrameHeader
from server.snapshot_adapter import SnapshotAdapter
from schemas.recovery import RecoveryStateSnapshot
from schemas.world_state import (
    ExecutionStateSnapshot,
    MemoryStateSnapshot,
    PerceptionStateSnapshot,
    PlanningStateSnapshot,
    RobotStateSnapshot,
    RuntimeStateSnapshot,
    SafetyStateSnapshot,
    TaskSnapshot,
    WorldStateSnapshot,
)
from webrtc.models import FrameCache


def _snapshot() -> WorldStateSnapshot:
    return WorldStateSnapshot(
        task=TaskSnapshot(task_id="task-1", instruction="go to apple", mode="NAV", state="active", command_id=8),
        mode="NAV",
        robot=RobotStateSnapshot(
            pose_xyz=(1.0, 2.0, 3.0),
            yaw_rad=0.25,
            frame_id=11,
            timestamp_ns=1234,
            source="unit_test",
            sensor_health={"observation_available": True, "batch_available": True},
            sensor_meta={"room_id": "dock"},
            capture_report={"sensor": "ok"},
        ),
        perception=PerceptionStateSnapshot(
            summary={"frame_id": 11},
            detector_backend="stub",
            detector_selected_reason="unit-test",
            detector_ready=True,
            detector_runtime_report={"ready": True},
            detection_count=1,
            tracked_detection_count=1,
            trajectory_point_count=2,
        ),
        memory=MemoryStateSnapshot(
            summary={"text_line_count": 2},
            object_count=2,
            place_count=1,
            semantic_rule_count=3,
            keyframe_count=4,
            scratchpad={"instruction": "go to apple"},
            memory_aware_task_active=True,
        ),
        planning=PlanningStateSnapshot(
            last_s2_result={"goal_version": 2},
            active_nav_plan={
                "plan_version": 4,
                "trajectory_point_count": 2,
                "trajectory_world": [[1.0, 2.0, 0.0], [1.5, 2.4, 0.0]],
            },
            plan_version=4,
            goal_version=2,
            traj_version=3,
            planner_mode="NAV",
            active_instruction="go to apple",
            route_state={"pixelGoal": [24, 18], "plannerControlMode": "trajectory", "plannerControlReason": "route_refresh"},
            planner_control_mode="trajectory",
            planner_control_reason="route_refresh",
            planner_yaw_delta_rad=0.12,
            system2_pixel_goal=[24, 18],
            stale_info={"planner_stale_sec": 0.4},
            global_route={"enabled": True, "active": True, "waypoint_index": 1, "waypoint_count": 3},
        ),
        execution=ExecutionStateSnapshot(
            last_command_decision={"source": "manual"},
            last_action_status={"state": "running"},
            active_overrides={"safety_override": False},
            locomotion_proposal_summary={
                "goal_distance_m": 1.5,
                "yaw_error_rad": 0.04,
                "command_vector": [0.12, -0.03, 0.2],
            },
            active_command_type="NAV_TO_POSE",
            active_target={"action_type": "NAV_TO_POSE", "target_track_id": "track-1"},
        ),
        safety=SafetyStateSnapshot(
            safe_stop=False,
            stale=True,
            timeout=False,
            sensor_unavailable=False,
            recovery_state=RecoveryStateSnapshot(
                current_state="REPLAN_PENDING",
                entered_at_ns=55,
                retry_count=1,
                backoff_until_ns=88,
                last_trigger_reason="trajectory_stale",
            ),
        ),
        runtime=RuntimeStateSnapshot(
            launch_mode="g1_view",
            viewer_publish=True,
            native_viewer="off",
            scene_preset="warehouse",
            show_depth=True,
            memory_store=True,
            detection_enabled=True,
            control_endpoint="tcp://127.0.0.1:5580",
            telemetry_endpoint="tcp://127.0.0.1:5581",
            shm_name="g1_view_frames",
            frame_available=True,
        ),
    )


def _frame() -> FrameCache:
    return FrameCache(
        seq=7,
        frame_header=FrameHeader(
            frame_id=11,
            timestamp_ns=1234,
            source="unit_test",
            width=32,
            height=24,
            robot_pose_xyz=(9.0, 9.0, 9.0),
            robot_yaw_rad=1.5,
            sim_time_s=4.5,
            metadata={"planner_overlay": {"plan_version": 999}},
        ),
        rgb_image=np.zeros((24, 32, 3), dtype=np.uint8),
        depth_image_m=np.full((24, 32), 1.5, dtype=np.float32),
        viewer_overlay={
            "detections": [
                {
                    "class_name": "apple",
                    "bbox_xyxy": [1, 2, 10, 12],
                    "track_id": "track-1",
                    "depth_m": 1.5,
                }
            ],
            "trajectory_pixels": [[10, 12], [15, 18]],
            "active_target": {"action_type": "LOCAL_SEARCH"},
        },
        last_frame_monotonic=1.0,
    )


def test_snapshot_adapter_preserves_legacy_runtime_contract() -> None:
    payload = SnapshotAdapter.to_legacy_runtime_payload(_snapshot())

    assert set(payload) == {"modes", "planner", "sensor", "perception", "memory", "transport"}
    assert payload["modes"]["executionMode"] == "NAV"
    assert payload["planner"]["planVersion"] == 4
    assert payload["planner"]["activeInstruction"] == "go to apple"
    assert payload["planner"]["routeState"]["pixelGoal"] == [24, 18]
    assert payload["planner"]["recoveryState"] == "REPLAN_PENDING"
    assert payload["planner"]["navTrajectoryWorld"] == [[1.0, 2.0, 0.0], [1.5, 2.4, 0.0]]
    assert payload["planner"]["navTrajectoryPointCount"] == 2
    assert payload["planner"]["commandVector"] == [0.12, -0.03, 0.2]
    assert payload["sensor"]["frameId"] == 11
    assert payload["sensor"]["stale"] is True
    assert payload["perception"]["detectorBackend"] == "stub"
    assert payload["memory"]["objectCount"] == 2
    assert payload["transport"]["viewerPublish"] is True


def test_snapshot_adapter_builds_dashboard_state_from_snapshot() -> None:
    state = SnapshotAdapter.to_dashboard_state(
        _snapshot(),
        processes=[{"name": "navdp", "state": "running"}],
        services={"navdp": {"status": "ok", "latencyMs": 18.0}, "dual": {"status": "ok", "latencyMs": 24.0}},
        session_state={"timestamp": 1.0, "active": True, "startedAt": 2.0, "config": {"viewerEnabled": True}, "lastEvent": {"message": "ok"}},
        transport_state={"frameSeq": 7, "frameAvailable": True, "frameAgeMs": 9.0},
        architecture={
            "gateway": {"state": "running"},
            "mainControlServer": {"state": "running"},
            "modules": {"nav": {"state": "running"}},
        },
        recent_logs=[{"message": "log"}],
        last_status={"state": "running"},
        detector_capability={"component": "detector", "status": "ready"},
        cognition_trace=[{"frameId": 11}],
        recovery_transitions=[{"from": "NORMAL", "to": "REPLAN_PENDING", "reason": "trajectory_stale", "timestamp": 1.0, "retryCount": 1}],
    )

    assert state["runtime"]["modes"]["executionMode"] == "NAV"
    assert state["runtime"]["lastStatusEvent"]["state"] == "running"
    assert state["runtime"]["navTrajectoryPointCount"] == 2
    assert state["runtime"]["commandVector"] == [0.12, -0.03, 0.2]
    assert state["perception"]["detectorCapability"]["status"] == "ready"
    assert state["transport"]["frameSeq"] == 7
    assert state["selectedTargetSummary"] == {
        "className": "",
        "trackId": "track-1",
        "source": "manual",
    }
    assert state["latencyBreakdown"] == {
        "frameAgeMs": 9.0,
        "perceptionLatencyMs": None,
        "memoryLatencyMs": None,
        "s2LatencyMs": 24.0,
        "navLatencyMs": 18.0,
        "locomotionLatencyMs": None,
    }
    assert state["cognitionTrace"] == [{"frameId": 11}]
    assert state["recoveryTransitions"][0]["to"] == "REPLAN_PENDING"


def test_snapshot_adapter_normalizes_target_and_latency_summaries() -> None:
    summary = SnapshotAdapter.selected_target_summary(_snapshot(), frame=_frame())
    latency = SnapshotAdapter.latency_breakdown(
        _snapshot(),
        services={"navdp": {"latencyMs": 18.0}, "dual": {"latencyMs": 24.0}},
        transport_state={"frameAgeMs": 9.0},
    )

    assert summary == {
        "className": "apple",
        "trackId": "track-1",
        "source": "perception",
        "bbox": [1, 2, 10, 12],
        "depthM": 1.5,
    }
    assert latency == {
        "frameAgeMs": 9.0,
        "perceptionLatencyMs": None,
        "memoryLatencyMs": None,
        "s2LatencyMs": 24.0,
        "navLatencyMs": 18.0,
        "locomotionLatencyMs": None,
    }


def test_snapshot_adapter_uses_world_state_for_webrtc_payloads() -> None:
    snapshot = _snapshot()
    frame = _frame()

    state_payload = SnapshotAdapter.to_webrtc_state_payload(snapshot, frame=frame, has_seen_frame=True, age_ms=5.0)
    frame_meta = SnapshotAdapter.to_webrtc_frame_meta(snapshot, frame=frame)

    assert state_payload["type"] == "snapshot"
    assert state_payload["robot_pose_xyz"] == [1.0, 2.0, 3.0]
    assert state_payload["planVersion"] == 4
    assert state_payload["active_command_type"] == "NAV_TO_POSE"
    assert state_payload["executionMode"] == "NAV"
    assert state_payload["activeInstruction"] == "go to apple"
    assert state_payload["system2PixelGoal"] == [24, 18]
    assert state_payload["recoveryState"] == "REPLAN_PENDING"

    assert frame_meta is not None
    assert frame_meta["robot_pose_xyz"] == [1.0, 2.0, 3.0]
    assert frame_meta["planVersion"] == 4
    assert frame_meta["stale"] is True
    assert frame_meta["activeTarget"]["target_track_id"] == "track-1"
    assert frame_meta["detections"][0]["class_name"] == "apple"
