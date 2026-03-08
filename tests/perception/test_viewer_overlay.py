from __future__ import annotations

import json
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
            "detector": "tensorrt_yoloe",
            "detector_selected_reason": "engine_ready",
        },
    )

    payload = build_viewer_overlay_payload(frame_result)

    assert payload == {
        "detector_backend": "tensorrt_yoloe",
        "detector_selected_reason": "engine_ready",
        "detections": [
            {
                "class_name": "apple",
                "confidence": 0.91,
                "bbox_xyxy": [10, 12, 40, 56],
                "track_id": "apple_0001",
                "depth_m": 1.2346,
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
