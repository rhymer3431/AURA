from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.messages import HealthPing
from ipc.messages import FrameHeader
from webrtc.models import (
    FrameCache,
    build_frame_meta_message,
    build_session_ready_message,
    build_snapshot_message,
    build_waiting_for_frame_message,
    frame_age_ms,
    ipc_message_event,
    is_frame_stale,
)


def _frame_cache() -> FrameCache:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    depth = np.full((24, 32), 1.5, dtype=np.float32)
    return FrameCache(
        seq=7,
        frame_header=FrameHeader(
            frame_id=11,
            timestamp_ns=1234,
            source="unit_test",
            width=32,
            height=24,
            robot_pose_xyz=(1.0, 2.0, 3.0),
            robot_yaw_rad=0.25,
            sim_time_s=4.5,
            metadata={
                "planner_overlay": {
                    "plan_version": 4,
                    "goal_version": 2,
                    "traj_version": 3,
                    "stale_sec": 0.4,
                    "planner_control_mode": "trajectory",
                    "planner_yaw_delta_rad": 0.12,
                    "interactive_phase": "task_active",
                    "interactive_command_id": 8,
                    "interactive_instruction": "go to apple",
                }
            },
        ),
        rgb_image=rgb,
        depth_image_m=depth,
        viewer_overlay={
            "detector_backend": "stub",
            "detections": [
                {
                    "class_name": "apple",
                    "confidence": 0.9,
                    "bbox_xyxy": [1, 2, 10, 12],
                    "track_id": "track-1",
                    "depth_m": 1.5,
                    "world_pose_xyz": [0.1, 0.2, 0.3],
                    "approach_yaw_rad": 0.4,
                }
            ],
            "trajectory_pixels": [[10, 12], [15, 18]],
            "system2_pixel_goal": [24, 18],
            "active_target": {
                "action_type": "NAV_TO_POSE",
                "target_track_id": "track-1",
                "nav_goal_pixel": [20, 22],
            },
        },
        last_frame_monotonic=time.monotonic(),
    )


def test_frame_messages_follow_contract() -> None:
    frame = _frame_cache()

    snapshot = build_snapshot_message(frame, active_command_type="LOCAL_SEARCH")
    meta = build_frame_meta_message(frame)
    ready = build_session_ready_message(
        session_id="abc123",
        track_roles=["rgb", "depth"],
        channel_labels=("state", "telemetry"),
    )

    assert snapshot["type"] == "snapshot"
    assert snapshot["seq"] == 7
    assert snapshot["detector_backend"] == "stub"
    assert snapshot["detection_count"] == 1
    assert snapshot["active_command_type"] == "NAV_TO_POSE"
    assert snapshot["has_depth"] is True
    assert snapshot["planVersion"] == 4
    assert snapshot["goalVersion"] == 2
    assert snapshot["trajVersion"] == 3
    assert snapshot["interactiveCommandId"] == 8
    assert snapshot["system2PixelGoal"] == [24, 18]

    assert meta["type"] == "frame_meta"
    assert meta["frame_id"] == 11
    assert meta["detections"][0]["class_name"] == "apple"
    assert meta["detections"][0]["bbox_xyxy"] == [1, 2, 10, 12]
    assert meta["trajectory_pixels"] == [[10, 12], [15, 18]]
    assert meta["trajectoryPixels"] == [[10, 12], [15, 18]]
    assert meta["system2PixelGoal"] == [24, 18]
    assert meta["active_target"]["target_track_id"] == "track-1"
    assert meta["activeTarget"]["target_track_id"] == "track-1"
    assert meta["interactiveInstruction"] == "go to apple"

    assert ready["sessionId"] == "abc123"
    assert ready["trackRoles"] == ["rgb", "depth"]


def test_waiting_and_ipc_helpers_are_json_safe() -> None:
    frame = _frame_cache()
    waiting = build_waiting_for_frame_message(age_ms=12.3456, has_seen_frame=True)
    event = ipc_message_event("health", HealthPing(component="bridge", details={"alive": True}))

    assert waiting == {"type": "waiting_for_frame", "age_ms": 12.346, "has_seen_frame": True}
    assert event["type"] == "health"
    assert event["component"] == "bridge"
    assert event["details"] == {"alive": True}

    old_frame = FrameCache(
        seq=frame.seq,
        frame_header=frame.frame_header,
        rgb_image=frame.rgb_image,
        depth_image_m=frame.depth_image_m,
        viewer_overlay=frame.viewer_overlay,
        last_frame_monotonic=time.monotonic() - 3.0,
    )
    assert frame_age_ms(frame) is not None
    assert is_frame_stale(old_frame, stale_after_sec=1.0) is True
