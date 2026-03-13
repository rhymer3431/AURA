from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from ipc.messages import FrameHeader
from runtime.supervisor import Supervisor, SupervisorConfig


def _synthetic_batch() -> IsaacObservationBatch:
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    rgb[24:72, 56:92, 0] = 255
    depth = np.full((96, 96), 1.5, dtype=np.float32)
    intrinsic = np.asarray([[96.0, 0.0, 48.0], [0.0, 96.0, 48.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return IsaacObservationBatch(
        frame_header=FrameHeader(
            frame_id=1,
            timestamp_ns=1_000_000_000,
            source="test_supervisor",
            width=96,
            height=96,
            camera_pose_xyz=(0.0, 0.0, 1.2),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            robot_pose_xyz=(0.0, 0.0, 0.0),
            robot_yaw_rad=0.0,
            sim_time_s=1.0,
            metadata={"target_class_hint": "apple", "color_hint": "red"},
        ),
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        sim_time_s=1.0,
        rgb_image=rgb,
        depth_image_m=depth,
        camera_intrinsic=intrinsic,
        capture_report={"rgb_source": "synthetic", "depth_source": "synthetic"},
    )


def test_supervisor_visible_target_task_emits_nav_to_pose_from_processed_frame() -> None:
    supervisor = Supervisor(
        config=SupervisorConfig(
            detector_model_path="artifacts/models/__missing__.engine",
            detector_device="",
        )
    )
    batch = _synthetic_batch()

    supervisor.process_frame(batch, publish=False)
    supervisor.submit_task("보이는 사과로 가", target_json={"target_mode": "goto_visible_object", "target_class": "apple"})
    command = supervisor.step(now=1.1, robot_pose=(0.0, 0.0, 0.0), publish=False)

    assert command is not None
    assert command.action_type == "NAV_TO_POSE"
    assert command.metadata["target_mode"] == "goto_visible_object"
    assert command.target_pose_xyz is not None
    assert float(command.metadata["raw_target_pose_xyz"][0]) > 0.0
    assert tuple(round(float(value), 4) for value in command.target_pose_xyz) == (0.0, 0.0, 0.0)


def test_supervisor_can_disable_memory_storage_while_processing_frames() -> None:
    supervisor = Supervisor(
        config=SupervisorConfig(
            detector_model_path="artifacts/models/__missing__.engine",
            detector_device="",
            memory_store=False,
        )
    )

    supervisor.process_frame(_synthetic_batch(), publish=False)

    assert supervisor.memory_service.spatial_store.objects == {}
    assert list(supervisor.memory_service.temporal_store) == []


def test_supervisor_can_skip_detection_while_processing_frames() -> None:
    supervisor = Supervisor(
        config=SupervisorConfig(
            detector_model_path="artifacts/models/__missing__.engine",
            detector_device="",
            skip_detection=True,
        )
    )

    enriched = supervisor.process_frame(_synthetic_batch(), publish=False)

    assert enriched.observations == []
    assert supervisor.memory_service.spatial_store.objects == {}
