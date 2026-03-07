from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services.memory_service import MemoryService
from perception.pipeline import PerceptionPipeline


def test_detector_results_flow_into_spatial_memory_association() -> None:
    pipeline = PerceptionPipeline()
    memory_service = MemoryService()
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    rgb[24:72, 28:68, 0] = 255
    depth = np.full((96, 96), 1.5, dtype=np.float32)
    intrinsic = np.asarray([[96.0, 0.0, 48.0], [0.0, 96.0, 48.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    frame = pipeline.process_frame(
        rgb_image=rgb,
        depth_image_m=depth,
        timestamp=10.0,
        camera_pose_xyz=(0.0, 0.0, 1.2),
        camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        camera_intrinsic=intrinsic,
        metadata={
            "target_class_hint": "apple",
            "color_hint": "red",
            "room_id": "kitchen",
            "robot_pose_xyz": [0.0, 0.0, 0.0],
            "robot_yaw_rad": 0.0,
            "frame_source": "synthetic",
            "capture_report": {"rgb_source": "synthetic"},
        },
    )
    results = memory_service.observe_objects(frame.observations)

    assert len(frame.observations) == 1
    assert len(results) == 1
    assert results[0].object_node.class_name == "apple"
    assert results[0].place_node.room_id == "kitchen"
    assert results[0].object_node.last_place_id == results[0].place_node.place_id
    assert "bearing_yaw_rad" in frame.observations[0].metadata
    assert frame.observations[0].metadata["frame_source"] == "synthetic"
