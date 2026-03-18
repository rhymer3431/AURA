from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.models import MemoryContextBundle, RetrievedMemoryLine, ScratchpadState
from server.dual_planner_service import DualPlannerService, GoalCache, S2Result, TrajectoryCache


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
        vlm_num_history=8,
        vlm_max_images_per_request=3,
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
    orchestrator = DualPlannerService(_args())

    orchestrator._finish_s2(
        S2Result(
            ok=True,
            mode="stop",
            stop=True,
            reason="arrived",
            source="llm",
            raw_text="STOP",
        ),
        finished_at=123.0,
        generation=0,
    )

    assert orchestrator.goal_cache is not None
    assert orchestrator.goal_cache.mode == "wait"
    assert orchestrator.goal_cache.stop is False
    assert "suppressed" in orchestrator.goal_cache.reason
    assert orchestrator.s2_stop_suppressed_count == 1
    snapshot = orchestrator.debug_state()
    assert snapshot["stats"]["last_s2_requested_stop"] is True
    assert snapshot["stats"]["last_s2_effective_stop"] is False
    assert snapshot["stats"]["last_s2_mode"] == "wait"
    assert snapshot["stats"]["last_s2_needs_requery"] is True


def test_stop_after_confirmed_trajectory_is_preserved() -> None:
    orchestrator = DualPlannerService(_args())
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
            mode="stop",
            stop=True,
            reason="arrived",
            source="llm",
            raw_text="STOP",
        ),
        finished_at=124.0,
        generation=0,
    )

    assert orchestrator.goal_cache is not None
    assert orchestrator.goal_cache.mode == "stop"
    assert orchestrator.goal_cache.stop is True
    assert orchestrator.s2_stop_suppressed_count == 0
    snapshot = orchestrator.debug_state()
    assert snapshot["stats"]["last_s2_requested_stop"] is True
    assert snapshot["stats"]["last_s2_effective_stop"] is True


def test_identical_goal_refresh_keeps_goal_version() -> None:
    orchestrator = DualPlannerService(_args())

    orchestrator._finish_s2(
        S2Result(
            ok=True,
            mode="pixel_goal",
            pixel_x=11,
            pixel_y=22,
            stop=False,
            reason="forward",
            source="llm",
            raw_text="22, 11",
        ),
        finished_at=123.0,
        generation=0,
    )

    assert orchestrator.goal_cache is not None
    first_version = orchestrator.goal_cache.version

    orchestrator._finish_s2(
        S2Result(
            ok=True,
            mode="pixel_goal",
            pixel_x=11,
            pixel_y=22,
            stop=False,
            reason="forward again",
            source="llm",
            raw_text="22, 11",
        ),
        finished_at=124.0,
        generation=0,
    )

    assert orchestrator.goal_cache is not None
    assert orchestrator.goal_cache.version == first_version
    assert orchestrator.goal_version == first_version
    assert orchestrator.goal_cache.reason == "forward again"


def test_step_skips_periodic_s2_until_first_trajectory() -> None:
    orchestrator = DualPlannerService(_args())
    orchestrator.initialized = True
    orchestrator.goal_version = 0
    orchestrator.traj_version = -1
    orchestrator.goal_cache = GoalCache(
        mode="pixel_goal",
        pixel_x=10,
        pixel_y=20,
        stop=False,
        yaw_delta_rad=None,
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
        memory_context=None,
        events={},
    )

    assert response["debug"]["called_s1"] is True
    assert response["debug"]["called_s2"] is False


def test_finish_s2_ignores_stale_generation_results() -> None:
    orchestrator = DualPlannerService(_args())
    orchestrator._generation = 1

    orchestrator._finish_s2(
        S2Result(
            ok=True,
            mode="pixel_goal",
            pixel_x=9,
            pixel_y=19,
            stop=False,
            reason="stale",
            source="llm",
            raw_text="19, 9",
        ),
        finished_at=125.0,
        generation=0,
    )

    assert orchestrator.goal_cache is None
    assert orchestrator.s2_calls == 0


def test_step_returns_yaw_delta_without_launching_s1() -> None:
    orchestrator = DualPlannerService(_args())
    orchestrator.initialized = True
    orchestrator.goal_version = 0
    orchestrator.goal_cache = GoalCache(
        mode="yaw_delta",
        pixel_x=None,
        pixel_y=None,
        stop=False,
        yaw_delta_rad=float(np.pi / 6.0),
        reason="←",
        version=0,
        updated_at=time.time(),
        source="llm",
    )
    orchestrator._s1_worker = lambda *args, **kwargs: None  # type: ignore[method-assign]
    orchestrator._s2_worker = lambda *args, **kwargs: None  # type: ignore[method-assign]

    response = orchestrator.step(
        image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.ones((8, 8), dtype=np.float32),
        step_id=2,
        cam_pos=np.zeros(3, dtype=np.float32),
        cam_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor_meta={},
        memory_context=None,
        events={},
    )

    assert response["trajectory_world"] == []
    assert response["planner_control"]["mode"] == "yaw_delta"
    assert response["planner_control"]["yaw_delta_rad"] == float(np.pi / 6.0)
    assert response["debug"]["called_s1"] is False


def test_step_routes_memory_context_only_to_s2() -> None:
    orchestrator = DualPlannerService(_args())
    orchestrator.initialized = True
    orchestrator.goal_version = 0
    orchestrator.last_s1_ts = 0.0
    orchestrator.last_s2_ts = 0.0
    orchestrator.goal_cache = GoalCache(
        mode="pixel_goal",
        pixel_x=10,
        pixel_y=20,
        stop=False,
        yaw_delta_rad=None,
        reason="goal",
        version=0,
        updated_at=time.time(),
        source="llm",
    )
    captured: dict[str, object] = {}
    memory_context = MemoryContextBundle(
        instruction="find apple",
        scratchpad=ScratchpadState(
            instruction="find apple",
            planner_mode="interactive",
            task_state="active",
        ),
        text_lines=[
            RetrievedMemoryLine(
                text="Apple seen in kitchen.",
                score=3.0,
                source_type="object_memory",
                entity_id="apple_0001",
            )
        ],
    )

    def _fake_prepare(events, memory_bundle):  # noqa: ANN001
        captured["events"] = dict(events)
        captured["memory_context"] = memory_bundle
        return SimpleNamespace(body={})

    def _fake_s1_worker(  # noqa: ANN001
        image_bgr,
        depth_m,
        pixel_goal,
        sensor_meta,
        cam_pos,
        cam_quat_wxyz,
        goal_version,
        generation,
    ):
        _ = image_bgr, depth_m, pixel_goal, cam_pos, cam_quat_wxyz, goal_version, generation
        captured["sensor_meta"] = dict(sensor_meta)

    orchestrator._prepare_s2_request = _fake_prepare  # type: ignore[method-assign]
    orchestrator._s2_worker = lambda request, generation: None  # type: ignore[method-assign]
    orchestrator._s1_worker = _fake_s1_worker  # type: ignore[method-assign]

    orchestrator.step(
        image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.ones((8, 8), dtype=np.float32),
        step_id=3,
        cam_pos=np.zeros(3, dtype=np.float32),
        cam_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        sensor_meta={"frame_source": "camera"},
        memory_context=memory_context,
        events={"force_s2": True},
    )

    deadline = time.time() + 0.5
    while "sensor_meta" not in captured and time.time() < deadline:
        time.sleep(0.01)

    assert captured["memory_context"] == memory_context
    assert captured["sensor_meta"] == {"frame_source": "camera"}
