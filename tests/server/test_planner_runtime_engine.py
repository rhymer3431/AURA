from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control.async_planners import PlannerInput, PlannerOutput
from ipc.messages import ActionCommand
from runtime.planning_session import ExecutionObservation, PlanningSession
from server.planner_runtime_engine import PlannerRuntimeEngine
from server.planner_runtime_state import PlannerRuntimeState


def _args(*, planner_mode: str = "interactive") -> Namespace:
    return Namespace(
        planner_mode=planner_mode,
        server_url="http://127.0.0.1:8888",
        system2_url="http://127.0.0.1:15801",
        instruction="head to dock",
        use_trajectory_z=False,
        plan_wait_timeout_sec=0.5,
        plan_interval_frames=1,
        s1_period_sec=0.2,
        s2_period_sec=0.0,
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
        internvla_goal_depth_window=3,
        internvla_goal_depth_min=0.1,
        internvla_goal_depth_max=6.0,
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


class _FakeSystem2Client:
    def __init__(self) -> None:
        self.session_id = "system2-nav"
        self.reset_calls: list[str] = []
        self.results: list[Namespace] = []

    def reset(self, instruction: str) -> None:
        self.reset_calls.append(str(instruction))

    def step_session(self, *, session_id: str, rgb, depth, stamp_s: float):  # noqa: ANN001
        _ = session_id, rgb, depth, stamp_s
        if not self.results:
            raise AssertionError("step_session called without a prepared result")
        return self.results.pop(0)


class _FakePlanner:
    def __init__(self) -> None:
        self.outputs: list[PlannerOutput | None] = []
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


def _make_session(*, mode: str) -> tuple[PlanningSession, _FakePlanner, _FakePlanner, _FakeNavDPClient, _FakeSystem2Client]:
    session = PlanningSession(_args(planner_mode=mode))
    session._intrinsic = np.asarray([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    session.navdp_client = _FakeNavDPClient()
    session.pointgoal_planner = _FakePlanner()
    session.nogoal_planner = _FakePlanner()
    session.system2_client = _FakeSystem2Client()
    return session, session.nogoal_planner, session.pointgoal_planner, session.navdp_client, session.system2_client


def _observation(frame_id: int):
    return ExecutionObservation(
        frame_id=frame_id,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        sensor_meta={"rgb_source": "fake", "depth_source": "fake"},
        cam_pos=np.zeros(3, dtype=np.float32),
        cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        intrinsic=np.asarray([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_planning_session_default_navdp_client_factory_uses_supported_keyword_names(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_create_inprocess_navdp_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("runtime.planning_session.create_inprocess_navdp_client", _fake_create_inprocess_navdp_client)

    session = PlanningSession(_args())
    result = session._default_navdp_client_factory(np.eye(3, dtype=np.float32), _args())

    assert result is not None
    assert captured["backend"] == "heuristic"
    assert captured["amp"] is False
    assert captured["tf32"] is False
    assert "backend_name" not in captured
    assert "use_amp" not in captured
    assert "allow_tf32" not in captured


def test_planner_runtime_engine_updates_nav_goal_from_system2_pixel_goal() -> None:
    session, _nogoal_planner, pointgoal_planner, navdp_client, system2_client = _make_session(mode="nav")
    state = PlannerRuntimeState(mode="nav")
    engine = PlannerRuntimeEngine(_args(planner_mode="nav"), transport=session, state=state)

    system2_client.results = [
        Namespace(decision_mode="pixel_goal", pixel_xy=(5, 4), text="5, 4"),
    ]
    pointgoal_planner.outputs = [
        PlannerOutput(
            plan_version=0,
            source_frame_id=1,
            trajectory_world=np.asarray([[0.4, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32),
            latency_ms=3.5,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    pointgoal_planner.status = (1, 0, "", 3.5)

    engine.start_nav_task("head to the dock")
    update = engine.plan_with_observation(
        _observation(1),
        action_command=_planner_managed_command(task_id="nav"),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert system2_client.reset_calls == ["head to the dock"]
    assert navdp_client.reset_calls == 1
    assert state.goal.system2_pixel_goal == [5, 4]
    assert update.goal_version == 0
    assert update.traj_version == 0
    assert update.trajectory_world.shape == (2, 3)
    assert len(pointgoal_planner.submitted) == 1
    submitted = pointgoal_planner.submitted[0]
    assert isinstance(submitted, PlannerInput)
    assert submitted.sensor_meta["robot_pose_xyz"] == [0.0, 0.0, 0.0]
    assert submitted.sensor_meta["robot_yaw_rad"] == 0.0


def test_planner_runtime_engine_uses_direct_action_overrides() -> None:
    session, _nogoal_planner, pointgoal_planner, _navdp_client, system2_client = _make_session(mode="nav")
    state = PlannerRuntimeState(mode="nav")
    engine = PlannerRuntimeEngine(_args(planner_mode="nav"), transport=session, state=state)

    system2_client.results = [
        Namespace(decision_mode="forward", pixel_xy=None, text="forward"),
    ]

    engine.start_nav_task("move toward the dock")
    update = engine.plan_with_observation(
        _observation(2),
        action_command=_planner_managed_command(task_id="nav"),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert update.planner_control_mode == "forward"
    assert update.stop is False
    assert len(pointgoal_planner.submitted) == 0
    assert update.trajectory_world.shape == (0, 3)


def test_planner_runtime_engine_interactive_path_promotes_pending_instruction_into_nav_task() -> None:
    session, nogoal_planner, pointgoal_planner, navdp_client, system2_client = _make_session(mode="interactive")
    state = PlannerRuntimeState(mode="interactive")
    engine = PlannerRuntimeEngine(_args(planner_mode="interactive"), transport=session, state=state)

    assert engine.activate_interactive_roaming("startup") is True
    assert navdp_client.reset_calls == 1

    system2_client.results = [
        Namespace(decision_mode="pixel_goal", pixel_xy=(5, 4), text="5, 4"),
    ]
    pointgoal_planner.outputs = [
        PlannerOutput(
            plan_version=0,
            source_frame_id=3,
            trajectory_world=np.asarray([[0.3, 0.0, 0.0], [0.5, 0.1, 0.0]], dtype=np.float32),
            latency_ms=4.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    pointgoal_planner.status = (1, 0, "", 4.0)

    command_id = engine.submit_interactive_instruction("go to the loading dock")
    update = engine.plan_with_observation(
        _observation(3),
        action_command=_planner_managed_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert command_id == 1
    assert system2_client.reset_calls == ["go to the loading dock"]
    assert navdp_client.reset_calls == 2
    assert nogoal_planner.reset_calls == 2
    assert update.interactive_phase == "task_active"
    assert update.interactive_command_id == 1
    assert update.interactive_instruction == "go to the loading dock"
    assert state.goal.system2_pixel_goal == [5, 4]
