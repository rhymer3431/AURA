from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from control.async_planners import DualPlannerOutput, PlannerOutput
from ipc.messages import ActionCommand
from runtime.planning_session import PlanningSession
from server.planner_runtime_engine import PlannerRuntimeEngine
from server.planner_runtime_state import PlannerRuntimeState


def _args(*, planner_mode: str = "interactive") -> Namespace:
    return Namespace(
        planner_mode=planner_mode,
        server_url="http://127.0.0.1:8888",
        dual_server_url="http://127.0.0.1:8890",
        instruction="head to dock",
        use_trajectory_z=False,
        plan_wait_timeout_sec=0.5,
        plan_interval_frames=1,
        dual_request_gap_frames=1,
        s1_period_sec=0.2,
        s2_period_sec=1.0,
        goal_ttl_sec=3.0,
        traj_ttl_sec=1.5,
        traj_max_stale_sec=4.0,
        timeout_sec=5.0,
        reset_timeout_sec=15.0,
        retry=1,
        stop_threshold=-3.0,
        navdp_backend="heuristic",
        navdp_checkpoint="",
        navdp_device="cpu",
        navdp_amp=False,
        navdp_amp_dtype="float16",
        navdp_tf32=False,
        global_map_image="",
        global_map_config="",
        global_waypoint_spacing_m=0.75,
        global_inflation_radius_m=0.25,
    )


def _planner_managed_command(task_id: str = "interactive") -> ActionCommand:
    return ActionCommand(action_type="LOCAL_SEARCH", task_id=task_id, metadata={"planner_managed": True})


class _FakeNavDPClient:
    def __init__(self) -> None:
        self.reset_calls = 0

    def navigator_reset(self, intrinsic, batch_size=1):  # noqa: ANN001
        _ = intrinsic, batch_size
        self.reset_calls += 1
        return "navdp"


class _FakeDualClient:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    def dual_reset(self, intrinsic, instruction, **kwargs):  # noqa: ANN001,ANN003
        _ = intrinsic, kwargs
        self.instructions.append(str(instruction))
        return Namespace(algo="dual", state={})


class _FakePlanner:
    def __init__(self) -> None:
        self.outputs: list[PlannerOutput | DualPlannerOutput | None] = []
        self.status = (0, 0, "", 0.0)
        self.reset_calls = 0
        self.submitted: list[object] = []

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


def _make_session(*, mode: str) -> tuple[PlanningSession, _FakePlanner, _FakePlanner, _FakeNavDPClient, _FakeDualClient]:
    session = PlanningSession(_args(planner_mode=mode))
    session._intrinsic = np.eye(3, dtype=np.float32)
    session.navdp_client = _FakeNavDPClient()
    session.pointgoal_planner = _FakePlanner()
    session.nogoal_planner = _FakePlanner()
    session.dual_planner = _FakePlanner()
    session._dual_client = _FakeDualClient()
    return session, session.nogoal_planner, session.dual_planner, session.navdp_client, session._dual_client


def _observation(session: PlanningSession, frame_id: int):
    return session.build_local_observation(
        frame_id=frame_id,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        camera_pose_xyz=(0.0, 0.0, 1.2),
        camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsic=np.eye(3, dtype=np.float32),
        sensor_meta={"rgb_source": "fake", "depth_source": "fake"},
    )


def test_planning_session_no_longer_exposes_task_lifecycle_apis() -> None:
    session = PlanningSession(_args())
    for name in (
        "start_dual_task",
        "submit_interactive_instruction",
        "cancel_interactive_task",
        "active_memory_instruction",
        "viewer_overlay_state",
    ):
        assert not hasattr(session, name)


def test_planner_runtime_engine_owns_interactive_state_transitions() -> None:
    session, nogoal_planner, dual_planner, navdp_client, dual_client = _make_session(mode="interactive")
    state = PlannerRuntimeState(mode="interactive")
    engine = PlannerRuntimeEngine(_args(planner_mode="interactive"), transport=session, state=state)
    assert engine.activate_interactive_roaming("startup") is True

    dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=2,
            trajectory_world=np.asarray([[0.3, 0.0, 0.0], [0.5, 0.1, 0.0]], dtype=np.float32),
            pixel_goal=np.asarray([240.0, 180.0], dtype=np.float32),
            stop=False,
            goal_version=3,
            traj_version=4,
            stale_sec=0.2,
            used_cached_traj=False,
            planner_control={"mode": "trajectory", "reason": "forward", "yaw_delta_rad": None},
            debug={},
            latency_ms=4.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    dual_planner.status = (1, 0, "", 4.0)

    command_id = engine.submit_interactive_instruction("go to the loading dock")
    update = engine.plan_with_observation(
        _observation(session, 2),
        action_command=_planner_managed_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert navdp_client.reset_calls == 1
    assert command_id == 1
    assert dual_client.instructions == ["go to the loading dock"]
    assert update.interactive_phase == "task_active"
    assert update.interactive_command_id == 1
    assert state.goal.system2_pixel_goal == [240, 180]
    assert engine.viewer_overlay_state()["system2_pixel_goal"] == [240, 180]
    assert len(nogoal_planner.submitted) == 0


def test_planner_runtime_engine_owns_dual_goal_versions() -> None:
    session, _nogoal_planner, dual_planner, _navdp_client, dual_client = _make_session(mode="dual")
    state = PlannerRuntimeState(mode="dual")
    engine = PlannerRuntimeEngine(_args(planner_mode="dual"), transport=session, state=state)

    dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=1,
            trajectory_world=np.asarray([[0.4, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32),
            pixel_goal=np.asarray([240.0, 180.0], dtype=np.float32),
            stop=False,
            goal_version=5,
            traj_version=7,
            stale_sec=0.1,
            used_cached_traj=False,
            planner_control={"mode": "trajectory", "reason": "forward", "yaw_delta_rad": None},
            debug={},
            latency_ms=3.5,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    dual_planner.status = (1, 0, "", 3.5)

    engine.start_dual_task("head to the dock")
    update = engine.plan_with_observation(
        _observation(session, 1),
        action_command=_planner_managed_command(task_id="dual"),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert dual_client.instructions == ["head to the dock"]
    assert update.goal_version == 5
    assert update.traj_version == 7
    assert update.trajectory_world.shape == (2, 3)
