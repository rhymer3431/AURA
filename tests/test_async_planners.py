from __future__ import annotations

from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control.async_planners import AsyncNoGoalPlanner, NoGoalPlannerInput
from inference.vlm import AsyncSystem2Planner, System2SessionConfig
from inference.vlm.system2_session import System2Result


class _EmptyTrajectoryClient:
    def nogoal_step(self, *, rgb_images, depth_images_m):  # noqa: ANN001
        _ = rgb_images, depth_images_m
        return SimpleNamespace(
            trajectory=np.zeros((0, 3), dtype=np.float32),
            all_trajectory=np.zeros((0, 0, 3), dtype=np.float32),
            all_values=np.zeros((0, 0), dtype=np.float32),
        )


class _SlowSystem2Session:
    def __init__(self) -> None:
        self.calls = 0
        self.config = System2SessionConfig(endpoint="http://127.0.0.1:15801")

    def step_session(self, *, rgb, depth, stamp_s: float):  # noqa: ANN001
        _ = rgb, depth, stamp_s
        self.calls += 1
        time.sleep(0.05)
        return System2Result(
            status="goal",
            uv_norm=np.asarray([0.5, 0.5], dtype=np.float32),
            text="pixel_goal:4,4",
            latency_ms=50.0,
            stamp_s=float(stamp_s),
            pixel_xy=np.asarray([4.0, 4.0], dtype=np.float32),
            decision_mode="pixel_goal",
        )


def test_async_nogoal_planner_records_empty_trajectory_failure() -> None:
    planner = AsyncNoGoalPlanner(client=_EmptyTrajectoryClient(), use_trajectory_z=False)
    planner.start()
    try:
        planner.submit(
            NoGoalPlannerInput(
                frame_id=1,
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                depth=np.ones((8, 8), dtype=np.float32),
                sensor_meta={},
                cam_pos=np.zeros(3, dtype=np.float32),
                cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        )

        deadline = time.time() + 1.0
        while time.time() < deadline:
            _success, failed, _error, _latency = planner.snapshot_status()
            if failed > 0:
                break
            time.sleep(0.01)

        success, failed, error, _latency = planner.snapshot_status()

        assert success == 0
        assert failed == 1
        assert "empty trajectory" in error
    finally:
        planner.stop()


def test_async_system2_planner_returns_latest_result_without_blocking_submit() -> None:
    planner = AsyncSystem2Planner(_SlowSystem2Session())
    planner.start()
    try:
        planner.submit(
            SimpleNamespace(
                frame_id=7,
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                depth=np.ones((8, 8), dtype=np.float32),
                stamp_s=1.25,
            )
        )

        latest = None
        deadline = time.time() + 1.0
        while time.time() < deadline:
            latest = planner.consume_latest(-1)
            if latest is not None:
                break
            time.sleep(0.01)

        assert latest is not None
        assert latest.source_frame_id == 7
        assert latest.result.decision_mode == "pixel_goal"
        assert latest.result.pixel_xy is not None
        assert planner.has_pending_work() is False
    finally:
        planner.stop()
