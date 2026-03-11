from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.models import ObsObject
from services.memory_service import MemoryService


def test_record_perception_frame_updates_scratchpad_and_keyframes(tmp_path: Path) -> None:
    memory_service = MemoryService(
        db_path=str(tmp_path / "memory.sqlite"),
        keyframe_dir=str(tmp_path / "keyframes"),
    )
    memory_service.set_planner_task(
        instruction="find apple in kitchen",
        planner_mode="interactive",
        task_state="active",
        task_id="interactive",
        command_id=3,
    )

    rgb = np.full((48, 48, 3), 200, dtype=np.uint8)
    observations = [
        ObsObject(
            class_name="apple",
            track_id="apple-1",
            pose=(2.0, 1.0, 0.0),
            timestamp=10.0,
            confidence=0.95,
            room_id="kitchen",
            metadata={
                "bbox_xyxy": [8, 8, 24, 24],
                "depth_m": 1.2,
                "bearing_yaw_rad": -0.4,
            },
        )
    ]

    results = memory_service.record_perception_frame(
        frame_id=1,
        rgb_image=rgb,
        observations=observations,
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        instruction="find apple in kitchen",
    )

    assert len(results) == 1
    assert len(memory_service.keyframes) == 1
    keyframe = next(iter(memory_service.keyframes.values()))
    assert Path(keyframe.image_path).exists()
    assert keyframe.crop_paths
    assert memory_service.scratchpad.checked_locations == ["kitchen"]
    assert "Observed apple" in memory_service.scratchpad.recent_hint

    bundle = memory_service.build_memory_context(
        instruction="find apple in kitchen",
        current_pose=(0.0, 0.0, 0.0),
        max_text_lines=3,
        max_keyframes=1,
    )

    assert bundle is not None
    assert len(bundle.text_lines) <= 3
    assert len(bundle.keyframes) == 1
    assert bundle.keyframes[0].keyframe_id == keyframe.keyframe_id
    assert bundle.crop_path != ""


def test_build_memory_context_applies_topk_and_query_scoring(tmp_path: Path) -> None:
    memory_service = MemoryService(
        db_path=str(tmp_path / "memory.sqlite"),
        keyframe_dir=str(tmp_path / "keyframes"),
    )
    rgb = np.full((32, 32, 3), 127, dtype=np.uint8)

    memory_service.record_perception_frame(
        frame_id=1,
        rgb_image=rgb,
        observations=[
            ObsObject(
                class_name="apple",
                track_id="apple-1",
                pose=(1.0, 0.0, 0.0),
                timestamp=100.0,
                confidence=0.95,
                room_id="kitchen",
                metadata={"bbox_xyxy": [2, 2, 12, 12], "depth_m": 1.0, "bearing_yaw_rad": -0.2},
            ),
            ObsObject(
                class_name="mug",
                track_id="mug-1",
                pose=(1.5, 0.2, 0.0),
                timestamp=100.0,
                confidence=0.60,
                room_id="kitchen",
                metadata={"bbox_xyxy": [14, 2, 24, 12], "depth_m": 1.4, "bearing_yaw_rad": 0.2},
            ),
            ObsObject(
                class_name="book",
                track_id="book-1",
                pose=(3.0, 0.5, 0.0),
                timestamp=100.0,
                confidence=0.99,
                room_id="office",
                metadata={"bbox_xyxy": [4, 16, 16, 28], "depth_m": 2.2, "bearing_yaw_rad": 0.4},
            ),
        ],
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        instruction="find apple in kitchen",
    )

    bundle = memory_service.build_memory_context(
        instruction="find apple in kitchen",
        current_pose=(0.0, 0.0, 0.0),
        max_text_lines=2,
        max_keyframes=2,
    )

    assert bundle is not None
    assert len(bundle.text_lines) == 2
    assert "apple" in bundle.text_lines[0].text.lower()
    assert all(line.text.strip() != "" for line in bundle.text_lines)
