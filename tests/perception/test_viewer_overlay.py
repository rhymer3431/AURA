from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.detectors.base import DetectionResult
from inference.trackers.simple_tracker import TrackedDetection
from perception.depth_projection import ProjectedDetection
from perception.pipeline import PerceptionFrameResult
from perception.viewer_overlay import build_viewer_overlay_payload


def test_importing_speaker_events_does_not_eager_import_detector_backends() -> None:
    for module_name in [
        "perception",
        "perception.speaker_events",
        "perception.pipeline",
        "inference.detectors",
        "inference.detectors.factory",
        "inference.detectors.ultralytics_yolo",
    ]:
        sys.modules.pop(module_name, None)

    module = importlib.import_module("perception.speaker_events")

    assert hasattr(module, "SpeakerEvent")
    assert "perception.pipeline" not in sys.modules
    assert "inference.detectors.ultralytics_yolo" not in sys.modules


def test_build_viewer_overlay_payload_is_json_safe_and_stable() -> None:
    tracked = [
        TrackedDetection(
            track_id="apple_0001",
            detection=DetectionResult(
                class_name="apple",
                confidence=0.91,
                bbox_xyxy=(10, 12, 40, 56),
            ),
        ),
        TrackedDetection(
            track_id="person_0002",
            detection=DetectionResult(
                class_name="person",
                confidence=0.87654321,
                bbox_xyxy=(60, 16, 96, 88),
            ),
        ),
    ]
    projected = [
        ProjectedDetection(
            class_name="apple",
            confidence=0.91,
            world_pose_xyz=(1.0, 2.0, 0.5),
            track_id="apple_0001",
            metadata={"depth_m": 1.23456, "ignored": np.asarray([1, 2, 3], dtype=np.int32)},
        )
    ]
    frame_result = PerceptionFrameResult(
        detections=[item.detection for item in tracked],
        tracked_detections=tracked,
        projected_detections=projected,
        observations=[],
        speaker_events=[],
        metadata={
            "detector": "ultralytics_yolo",
            "detector_selected_reason": "ultralytics_backend_ready",
        },
    )

    payload = build_viewer_overlay_payload(frame_result)

    assert payload == {
        "detector_backend": "ultralytics_yolo",
        "detector_selected_reason": "ultralytics_backend_ready",
        "detections": [
            {
                "class_name": "apple",
                "confidence": 0.91,
                "bbox_xyxy": [10, 12, 40, 56],
                "track_id": "apple_0001",
                "depth_m": 1.2346,
                "world_pose_xyz": [1.0, 2.0, 0.5],
            },
            {
                "class_name": "person",
                "confidence": 0.876543,
                "bbox_xyxy": [60, 16, 96, 88],
                "track_id": "person_0002",
            },
        ],
    }
    assert json.loads(json.dumps(payload)) == payload


def test_build_viewer_overlay_payload_projects_planner_trajectory_to_pixels() -> None:
    frame_result = PerceptionFrameResult(
        detections=[],
        tracked_detections=[],
        projected_detections=[],
        observations=[],
        speaker_events=[],
        metadata={
            "detector": "stub",
            "detector_selected_reason": "unit_test",
            "planner_overlay": {
                "trajectory_world": [
                    [0.0, 0.0, 2.0],
                    [0.5, 0.0, 2.0],
                    [1.0, 0.0, 2.0],
                ],
                "plan_version": 7,
                "goal_version": 3,
                "traj_version": 5,
            },
        },
    )

    payload = build_viewer_overlay_payload(
        frame_result,
        camera_intrinsic=np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        camera_pose_xyz=(0.0, 0.0, 0.0),
        camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    assert payload["trajectory_pixels"] == [[50, 50], [75, 50], [100, 50]]
    assert payload["trajectory_point_count"] == 3
    assert payload["plan_version"] == 7
    assert payload["goal_version"] == 3
    assert payload["traj_version"] == 5


def test_build_viewer_overlay_payload_includes_active_target_debug() -> None:
    frame_result = PerceptionFrameResult(
        detections=[],
        tracked_detections=[],
        projected_detections=[],
        observations=[],
        speaker_events=[],
        metadata={
            "detector": "stub",
            "detector_selected_reason": "unit_test",
            "active_command_overlay": {
                "action_type": "NAV_TO_POSE",
                "target_mode": "goto_visible_object",
                "target_class": "apple",
                "target_track_id": "apple_0001",
                "pose_source": "filtered_track",
                "raw_target_pose_xyz": [0.0, 0.0, 2.0],
                "filtered_target_pose_xyz": [0.5, 0.0, 2.0],
                "nav_goal_pose_xyz": [1.0, 0.0, 2.0],
                "approach_yaw_rad": 0.0,
                "track_age_sec": 0.12,
            },
        },
    )

    payload = build_viewer_overlay_payload(
        frame_result,
        camera_intrinsic=np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        camera_pose_xyz=(0.0, 0.0, 0.0),
        camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    assert payload["active_target"]["action_type"] == "NAV_TO_POSE"
    assert payload["active_target"]["target_mode"] == "goto_visible_object"
    assert payload["active_target"]["raw_target_pixel"] == [50, 50]
    assert payload["active_target"]["filtered_target_pixel"] == [75, 50]
    assert payload["active_target"]["nav_goal_pixel"] == [100, 50]
