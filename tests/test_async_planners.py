from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from control.async_planners import AsyncDualPlanner, AsyncNoGoalPlanner, DualPlannerInput, NoGoalPlannerInput


class _EmptyTrajectoryClient:
    def nogoal_step(self, *, rgb_images, depth_images_m):  # noqa: ANN001
        _ = rgb_images, depth_images_m
        return SimpleNamespace(
            trajectory=np.zeros((0, 3), dtype=np.float32),
            all_trajectory=np.zeros((0, 0, 3), dtype=np.float32),
            all_values=np.zeros((0, 0), dtype=np.float32),
        )


def test_async_nogoal_planner_records_empty_trajectory_failure():
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


class _TransientWaitingDualClient:
    def dual_step(
        self,
        *,
        rgb_image,
        depth_image_m,
        step_id,
        cam_pos,
        cam_quat_wxyz,
        sensor_meta=None,
        events=None,
    ):  # noqa: ANN001
        _ = rgb_image, depth_image_m, step_id, cam_pos, cam_quat_wxyz, sensor_meta, events
        return SimpleNamespace(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            pixel_goal=None,
            stop=False,
            goal_version=-1,
            traj_version=-1,
            used_cached_traj=True,
            stale_sec=-1.0,
            debug={
                "called_s2": True,
                "called_s1": False,
                "s2_inflight": True,
                "s1_inflight": False,
                "force_s2_pending": False,
            },
        )


def test_async_dual_planner_treats_initial_empty_response_as_waiting():
    planner = AsyncDualPlanner(client=_TransientWaitingDualClient())
    planner.start()
    try:
        planner.submit(
            DualPlannerInput(
                frame_id=1,
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                depth=np.ones((8, 8), dtype=np.float32),
                sensor_meta={},
                cam_pos=np.zeros(3, dtype=np.float32),
                cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                events={},
            )
        )

        deadline = time.time() + 1.0
        while time.time() < deadline:
            success, failed, error, _latency = planner.snapshot_status()
            if error != "":
                break
            time.sleep(0.01)

        latest = planner.consume_latest(-1)
        success, failed, error, _latency = planner.snapshot_status()

        assert latest is None
        assert success == 0
        assert failed == 0
        assert "dual_step returned no active trajectory" in error
    finally:
        planner.stop()
