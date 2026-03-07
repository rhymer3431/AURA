from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services.dual_orchestrator import DualOrchestrator, GoalCache, S2Result, TrajectoryCache


def _args() -> Namespace:
    return Namespace(
        navdp_url="http://127.0.0.1:8888",
        vlm_url="http://127.0.0.1:8080",
        vlm_model="InternVLA-N1-System2.Q4_K_M.gguf",
        vlm_temperature=0.2,
        vlm_top_k=40,
        vlm_top_p=0.95,
        vlm_min_p=0.05,
        vlm_repeat_penalty=1.1,
        s2_mode="auto",
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


def test_initial_stop_is_suppressed_until_first_trajectory() -> None:
    orchestrator = DualOrchestrator(_args())

    orchestrator._finish_s2(
        S2Result(
            ok=True,
            pixel_x=10,
            pixel_y=20,
            stop=True,
            reason="arrived",
            source="llm",
            raw_text='{"pixel_x":10,"pixel_y":20,"stop":true,"reason":"arrived"}',
        ),
        finished_at=123.0,
    )

    assert orchestrator.goal_cache is not None
    assert orchestrator.goal_cache.stop is False
    assert "suppressed" in orchestrator.goal_cache.reason
    assert orchestrator.s2_stop_suppressed_count == 1
    snapshot = orchestrator.debug_state()
    assert snapshot["stats"]["last_s2_requested_stop"] is True
    assert snapshot["stats"]["last_s2_effective_stop"] is False


def test_stop_after_confirmed_trajectory_is_preserved() -> None:
    orchestrator = DualOrchestrator(_args())
    orchestrator.traj_version = 0
    orchestrator.traj_cache = TrajectoryCache(
        trajectory_world=np.zeros((2, 3), dtype=np.float32),
        version=0,
        goal_version=0,
        updated_at=1.0,
        latency_ms=10.0,
    )

    orchestrator._finish_s2(
        S2Result(
            ok=True,
            pixel_x=10,
            pixel_y=20,
            stop=True,
            reason="arrived",
            source="llm",
            raw_text='{"pixel_x":10,"pixel_y":20,"stop":true,"reason":"arrived"}',
        ),
        finished_at=124.0,
    )

    assert orchestrator.goal_cache is not None
    assert orchestrator.goal_cache.stop is True
    assert orchestrator.s2_stop_suppressed_count == 0
    snapshot = orchestrator.debug_state()
    assert snapshot["stats"]["last_s2_requested_stop"] is True
    assert snapshot["stats"]["last_s2_effective_stop"] is True


def test_identical_goal_refresh_keeps_goal_version() -> None:
    orchestrator = DualOrchestrator(_args())

    orchestrator._finish_s2(
        S2Result(
            ok=True,
            pixel_x=11,
            pixel_y=22,
            stop=False,
            reason="forward",
            source="llm",
            raw_text='{"pixel_x":11,"pixel_y":22,"stop":false,"reason":"forward"}',
        ),
        finished_at=123.0,
    )

    assert orchestrator.goal_cache is not None
    first_version = orchestrator.goal_cache.version

    orchestrator._finish_s2(
        S2Result(
            ok=True,
            pixel_x=11,
            pixel_y=22,
            stop=False,
            reason="forward again",
            source="llm",
            raw_text='{"pixel_x":11,"pixel_y":22,"stop":false,"reason":"forward again"}',
        ),
        finished_at=124.0,
    )

    assert orchestrator.goal_cache is not None
    assert orchestrator.goal_cache.version == first_version
    assert orchestrator.goal_version == first_version
    assert orchestrator.goal_cache.reason == "forward again"


def test_step_skips_periodic_s2_until_first_trajectory() -> None:
    orchestrator = DualOrchestrator(_args())
    orchestrator.initialized = True
    orchestrator.goal_version = 0
    orchestrator.traj_version = -1
    orchestrator.goal_cache = GoalCache(
        pixel_x=10,
        pixel_y=20,
        stop=False,
        reason="goal",
        version=0,
        updated_at=0.0,
        source="llm",
    )
    orchestrator.traj_cache = None
    orchestrator.last_s1_ts = 0.0
    orchestrator.last_s2_ts = 0.0
    orchestrator._s1_worker = lambda *args, **kwargs: None  # type: ignore[method-assign]
    orchestrator._s2_worker = lambda *args, **kwargs: None  # type: ignore[method-assign]

    response = orchestrator.step(
        image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.ones((8, 8), dtype=np.float32),
        step_id=1,
        cam_pos=np.zeros(3, dtype=np.float32),
        cam_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor_meta={},
        events={},
    )

    assert response["debug"]["called_s1"] is True
    assert response["debug"]["called_s2"] is False
