from __future__ import annotations

from argparse import Namespace

import numpy as np

from control.async_planners import DualPlannerOutput, PlannerOutput
from runtime.planning import PlannerSession


def _args() -> Namespace:
    return Namespace(
        planner_mode="interactive",
        server_url="http://127.0.0.1:8888",
        dual_server_url="http://127.0.0.1:8890",
        instruction="",
        goal_x=None,
        goal_y=None,
        goal_tolerance_m=0.4,
        spawn_demo_object=False,
        demo_object_x=2.0,
        demo_object_y=0.0,
        demo_object_size_m=0.25,
        object_stop_radius_m=0.8,
        plan_interval_frames=1,
        dual_request_gap_frames=1,
        safety_timeout_sec=20.0,
        s1_period_sec=0.2,
        s2_period_sec=1.0,
        goal_ttl_sec=3.0,
        traj_ttl_sec=1.5,
        traj_max_stale_sec=4.0,
        strict_d455=False,
        force_runtime_camera=False,
        use_trajectory_z=False,
        image_width=32,
        image_height=32,
        depth_max_m=5.0,
        timeout_sec=5.0,
        reset_timeout_sec=15.0,
        retry=1,
        stop_threshold=-3.0,
        startup_updates=0,
        log_interval=30,
        interactive_prompt="nl>",
        interactive_idle_log_interval=120,
        cmd_max_vx=0.5,
        cmd_max_vy=0.3,
        cmd_max_wz=0.8,
        lookahead_distance_m=0.6,
        heading_slowdown_rad=0.6,
        traj_stale_timeout_sec=1.5,
        cmd_accel_limit=1.0,
        cmd_yaw_accel_limit=1.5,
    )


class _FakeSensor:
    def __init__(self) -> None:
        self.intrinsic = np.eye(3, dtype=np.float32)

    def capture_rgbd_with_meta(self, _unused):  # noqa: ANN001
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        return rgb, depth, {"rgb_source": "fake", "depth_source": "fake"}

    def get_rgb_camera_pose_world(self):
        return np.zeros(3, dtype=np.float32), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class _FakeNavDPClient:
    def __init__(self) -> None:
        self.reset_calls = 0

    def navigator_reset(self, intrinsic, batch_size=1):  # noqa: ANN001
        assert batch_size == 1
        assert np.asarray(intrinsic).shape == (3, 3)
        self.reset_calls += 1
        return "navdp"


class _FakeDualClient:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    def dual_reset(
        self,
        intrinsic,
        instruction,
        *,
        navdp_url,
        s1_period_sec,
        s2_period_sec,
        goal_ttl_sec,
        traj_ttl_sec,
        traj_max_stale_sec,
    ):  # noqa: ANN001
        assert np.asarray(intrinsic).shape == (3, 3)
        assert navdp_url == "http://127.0.0.1:8888"
        assert s1_period_sec == 0.2
        assert s2_period_sec == 1.0
        assert goal_ttl_sec == 3.0
        assert traj_ttl_sec == 1.5
        assert traj_max_stale_sec == 4.0
        self.instructions.append(str(instruction))
        return Namespace(algo="dual", state={})


class _FakePlanner:
    def __init__(self) -> None:
        self.outputs: list[PlannerOutput | DualPlannerOutput | None] = []
        self.status = (0, 0, "", 0.0)
        self.submitted: list[object] = []
        self.reset_calls = 0

    def start(self) -> None:
        return None

    def stop(self, timeout_sec: float = 2.0) -> None:
        _ = timeout_sec

    def reset_state(self) -> None:
        self.reset_calls += 1

    def submit(self, planner_input) -> None:  # noqa: ANN001
        self.submitted.append(planner_input)

    def consume_latest(self, last_seen_version: int):  # noqa: ANN001
        _ = last_seen_version
        if not self.outputs:
            return None
        return self.outputs.pop(0)

    def snapshot_status(self):
        return self.status


def _make_session() -> tuple[PlannerSession, _FakePlanner, _FakePlanner, _FakeNavDPClient, _FakeDualClient]:
    session = PlannerSession(_args())
    session.sensor = _FakeSensor()
    nogoal_planner = _FakePlanner()
    dual_planner = _FakePlanner()
    nogoal_client = _FakeNavDPClient()
    dual_client = _FakeDualClient()
    session.nogoal_planner = nogoal_planner
    session.dual_planner = dual_planner
    session._nogoal_client = nogoal_client
    session._dual_client = dual_client
    assert session._activate_roaming("test startup") is True
    return session, nogoal_planner, dual_planner, nogoal_client, dual_client


def test_interactive_roaming_uses_nogoal_planner():
    session, nogoal_planner, _dual_planner, nogoal_client, _dual_client = _make_session()
    nogoal_planner.outputs = [
        PlannerOutput(
            plan_version=0,
            source_frame_id=1,
            trajectory_world=np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            latency_ms=3.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    nogoal_planner.status = (1, 0, "", 3.0)

    update = session.update(1)

    assert nogoal_client.reset_calls == 1
    assert len(nogoal_planner.submitted) == 1
    assert update.interactive_phase == "roaming"
    assert update.trajectory_world.shape == (2, 3)


def test_interactive_command_switches_to_task_mode_and_resets_dual():
    session, _nogoal_planner, dual_planner, _nogoal_client, dual_client = _make_session()
    dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=2,
            trajectory_world=np.asarray([[0.3, 0.0, 0.0], [0.5, 0.1, 0.0]], dtype=np.float32),
            stop=False,
            goal_version=3,
            traj_version=4,
            stale_sec=0.2,
            used_cached_traj=False,
            debug={},
            latency_ms=4.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    dual_planner.status = (1, 0, "", 4.0)
    command_id = session.submit_interactive_instruction("go to the loading dock")

    update = session.update(2)

    assert command_id == 1
    assert dual_client.instructions == ["go to the loading dock"]
    assert update.interactive_phase == "task_active"
    assert update.interactive_command_id == 1
    assert update.goal_version == 3
    assert update.traj_version == 4


def test_interactive_completion_returns_to_roaming():
    session, nogoal_planner, dual_planner, nogoal_client, dual_client = _make_session()
    dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=1,
            trajectory_world=np.asarray([[0.3, 0.0, 0.0], [0.5, 0.1, 0.0]], dtype=np.float32),
            stop=False,
            goal_version=1,
            traj_version=2,
            stale_sec=0.1,
            used_cached_traj=False,
            debug={},
            latency_ms=4.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        ),
        DualPlannerOutput(
            plan_version=1,
            source_frame_id=2,
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            stop=True,
            goal_version=2,
            traj_version=3,
            stale_sec=0.0,
            used_cached_traj=False,
            debug={},
            latency_ms=4.0,
            successful_calls=2,
            failed_calls=0,
            last_error="",
        ),
    ]
    dual_planner.status = (2, 0, "", 4.0)
    session.submit_interactive_instruction("inspect the pallet")

    first_update = session.update(1)
    second_update = session.update(2)

    assert first_update.interactive_phase == "task_active"
    assert second_update.interactive_phase == "roaming"
    assert second_update.interactive_command_id == -1
    assert dual_client.instructions == ["inspect the pallet"]
    assert nogoal_client.reset_calls == 2

    nogoal_planner.outputs = [
        PlannerOutput(
            plan_version=0,
            source_frame_id=3,
            trajectory_world=np.asarray([[0.1, 0.0, 0.0], [0.2, -0.1, 0.0]], dtype=np.float32),
            latency_ms=2.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    nogoal_planner.status = (1, 0, "", 2.0)

    roaming_update = session.update(3)

    assert roaming_update.interactive_phase == "roaming"
    assert roaming_update.trajectory_world.shape == (2, 3)


def test_interactive_new_instruction_preempts_active_task():
    session, _nogoal_planner, dual_planner, _nogoal_client, dual_client = _make_session()
    dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=1,
            trajectory_world=np.asarray([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32),
            stop=False,
            goal_version=1,
            traj_version=1,
            stale_sec=0.1,
            used_cached_traj=False,
            debug={},
            latency_ms=4.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        ),
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=2,
            trajectory_world=np.asarray([[0.0, 0.2, 0.0], [0.0, 0.4, 0.0]], dtype=np.float32),
            stop=False,
            goal_version=2,
            traj_version=2,
            stale_sec=0.1,
            used_cached_traj=False,
            debug={},
            latency_ms=4.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        ),
    ]
    dual_planner.status = (1, 0, "", 4.0)

    first_id = session.submit_interactive_instruction("go forward")
    first_update = session.update(1)
    second_id = session.submit_interactive_instruction("turn left and move")
    second_update = session.update(2)

    assert first_id == 1
    assert second_id == 2
    assert first_update.interactive_command_id == 1
    assert second_update.interactive_phase == "task_active"
    assert second_update.interactive_command_id == 2
    assert dual_client.instructions == ["go forward", "turn left and move"]


def test_interactive_cancel_returns_to_roaming():
    session, _nogoal_planner, dual_planner, nogoal_client, _dual_client = _make_session()
    dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=1,
            trajectory_world=np.asarray([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32),
            stop=False,
            goal_version=1,
            traj_version=1,
            stale_sec=0.1,
            used_cached_traj=False,
            debug={},
            latency_ms=4.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    dual_planner.status = (1, 0, "", 4.0)
    session.submit_interactive_instruction("go forward")
    task_update = session.update(1)

    assert task_update.interactive_phase == "task_active"
    assert session.cancel_interactive_task() is True

    cancel_update = session.update(2)

    assert cancel_update.interactive_phase == "roaming"
    assert cancel_update.interactive_command_id == -1
    assert nogoal_client.reset_calls == 2
