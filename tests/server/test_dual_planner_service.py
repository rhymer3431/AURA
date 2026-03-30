from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemas.workers import NavRequest, NavResult, inherit_worker_metadata, stamp_worker_metadata
from server.dual_planner_service import DualPlannerService


def _args() -> Namespace:
    return Namespace(
        navdp_url="http://127.0.0.1:8888",
        vlm_url="http://127.0.0.1:8080",
        vlm_model="mock-vlm",
        vlm_temperature=0.2,
        vlm_top_k=40,
        vlm_top_p=0.95,
        vlm_min_p=0.05,
        vlm_repeat_penalty=1.1,
        vlm_num_history=8,
        vlm_max_images_per_request=3,
        s2_mode="mock",
        s1_period_sec=0.2,
        s2_period_sec=1.0,
        goal_ttl_sec=3.0,
        traj_ttl_sec=1.5,
        traj_max_stale_sec=4.0,
        navdp_timeout_sec=5.0,
        navdp_reset_timeout_sec=15.0,
        vlm_timeout_sec=35.0,
        s2_failure_backoff_max_sec=30.0,
        stop_threshold=-3.0,
        use_trajectory_z=False,
        debug_log=False,
    )


def test_s1_worker_preserves_zero_goal_version() -> None:
    service = DualPlannerService(_args())
    service._generation = 1
    service._active_task_id = "dual:1"
    service.goal_version = 0
    service.s1_inflight = True

    request = NavRequest(
        metadata=stamp_worker_metadata(
            source="test.dual_planner_service",
            task_id="dual:1",
            frame_id=7,
            goal_version=0,
        ),
        image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.ones((8, 8), dtype=np.float32),
        pixel_goal=(4, 5),
        sensor_meta={},
        cam_pos=np.zeros(3, dtype=np.float32),
        cam_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    def _fake_call_s1(req: NavRequest) -> NavResult:
        return NavResult(
            metadata=inherit_worker_metadata(req.metadata, source="test.dual_planner_service.fake_s1"),
            trajectory_world=np.asarray([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32),
            latency_ms=1.0,
        )

    service._call_s1 = _fake_call_s1  # type: ignore[method-assign]

    service._s1_worker(request, generation=1)

    assert service.traj_cache is not None
    assert service.traj_cache.goal_version == 0
    assert service.traj_cache.version == 0
    assert service.s1_success == 1
    assert service.s1_inflight is False
