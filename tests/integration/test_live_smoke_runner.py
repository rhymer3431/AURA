from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.sensors.frame_source import build_synthetic_frame_sample
from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from memory.models import ObsObject
from runtime.isaac_launch_modes import LaunchModeAvailability, recommend_mode_for_failure, select_launch_mode
from runtime.live_smoke_runner import LiveSmokeRunner


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        launch_mode="standalone_python",
        headless=True,
        diagnostics_path=str(tmp_path / "diagnostics.json"),
        artifacts_dir=str(tmp_path / "artifacts"),
        assets_root="/Isaac",
        d455_asset_path="",
        d455_prim_path="/World/realsense_d455",
        image_width=96,
        image_height=96,
        depth_max_m=5.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=0.0,
        startup_updates=1,
        app_bootstrap_timeout_sec=30.0,
        stage_ready_timeout_sec=15.0,
        sensor_init_timeout_sec=15.0,
        first_frame_timeout_sec=0.0,
        scene_usd=None,
        env_url="/Isaac/Environments/Simple_Warehouse/warehouse.usd",
        scene_prim_path="/World/Environment",
        scene_translate=(0.0, 0.0, 0.0),
        force_runtime_camera=False,
        _argv=["--mode", "smoke"],
    )


class _EmptyFrameSource:
    def read(self):
        return None

    def report(self):
        return type("Report", (), {"notice": "frame timeout"})()


class _ObservationSupervisor:
    def __init__(self, *, memory_service) -> None:
        self.memory_service = memory_service

    def process_frame(self, batch: IsaacObservationBatch, *, publish: bool = False) -> IsaacObservationBatch:  # noqa: ARG002
        observation = ObsObject(
            class_name="apple",
            pose=(1.0, 0.0, 0.8),
            timestamp=batch.frame_header.timestamp_ns / 1.0e9,
            confidence=0.9,
            track_id="apple_track",
            place_id="place_kitchen",
            room_id="kitchen",
            metadata={"frame_source": batch.frame_header.source},
        )
        self.memory_service.observe_objects([observation])
        return IsaacObservationBatch(
            frame_header=batch.frame_header,
            robot_pose_xyz=batch.robot_pose_xyz,
            robot_yaw_rad=batch.robot_yaw_rad,
            sim_time_s=batch.sim_time_s,
            rgb_image=batch.rgb_image,
            depth_image_m=batch.depth_image_m,
            camera_intrinsic=batch.camera_intrinsic,
            observations=[observation],
            speaker_events=list(batch.speaker_events),
            capture_report=dict(batch.capture_report),
        )


def test_live_smoke_runner_distinguishes_first_frame_timeout(tmp_path: Path) -> None:
    runner = LiveSmokeRunner(_args(tmp_path))

    try:
        runner._wait_for_first_sample(_EmptyFrameSource())  # noqa: SLF001
    except TimeoutError:
        pass
    else:
        raise AssertionError("TimeoutError was expected for an empty frame source.")

    payload = Path(tmp_path / "diagnostics.json").read_text(encoding="utf-8")
    assert "first_rgb_frame_ready" in payload
    assert "timed out waiting for first frame" in payload


def test_live_smoke_runner_process_sample_reaches_memory_update(tmp_path: Path) -> None:
    args = _args(tmp_path)
    runner = LiveSmokeRunner(args, supervisor_factory=lambda *, memory_service: _ObservationSupervisor(memory_service=memory_service))
    sample = build_synthetic_frame_sample(
        frame_id=1,
        scene="apple",
        source_name="live_smoke_test",
        room_id="kitchen",
        width=96,
        height=96,
    )
    sample.metadata["capture_report"] = {"rgb_source": "d455_rgb_camera", "depth_source": "d455_depth_camera"}
    setattr(sample, "_pose_ready", True)

    metrics = runner._process_sample(sample)  # noqa: SLF001

    assert metrics["frame_received"] is True
    assert metrics["observation_batch_processed"] is True
    assert metrics["memory_updated"] is True
    assert metrics["observation_count"] == 1


def test_live_smoke_launch_mode_selection_and_recommendation() -> None:
    selection = select_launch_mode("auto", availability=LaunchModeAvailability(standalone_available=False, editor_available=True))
    assert selection.selected_mode == "full_app_attach"
    assert recommend_mode_for_failure(selected_mode="standalone_python", failure_phase="simulation_app_created") == "full_app_attach"
