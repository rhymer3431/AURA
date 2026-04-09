from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control.async_planners import PlannerInput, PlannerOutput
from inference.vlm import AsyncSystem2Output
from inference.vlm.system2_session import System2Result
from systems.transport.messages import ActionCommand
from runtime.planning_session import ExecutionObservation, PlanningSession
from server.planner_runtime_engine import PlannerRuntimeEngine
from server.planner_runtime_state import PlannerRuntimeState


def _args(**overrides: object) -> Namespace:
    defaults: dict[str, object] = {
        "planner_mode": "interactive",
        "server_url": "http://127.0.0.1:8888",
        "system2_url": "http://127.0.0.1:15801",
        "instruction": "head to dock",
        "nav_instruction_language": "auto",
        "use_trajectory_z": False,
        "plan_wait_timeout_sec": 0.5,
        "plan_interval_frames": 1,
        "s1_period_sec": 0.2,
        "s2_period_sec": 0.0,
        "goal_ttl_sec": 3.0,
        "traj_ttl_sec": 1.5,
        "traj_max_stale_sec": 4.0,
        "timeout_sec": 5.0,
        "reset_timeout_sec": 15.0,
        "retry": 1,
        "stop_threshold": -3.0,
        "goal_tolerance_m": 0.4,
        "navdp_backend": "heuristic",
        "navdp_checkpoint": "",
        "navdp_device": "cpu",
        "navdp_amp": False,
        "navdp_amp_dtype": "float16",
        "navdp_tf32": False,
        "global_map_image": "",
        "global_map_config": "",
        "global_waypoint_spacing_m": 0.75,
        "global_inflation_radius_m": 0.25,
        "navdp_replan_hz": 3.0,
        "navdp_plan_timeout": 1.5,
        "navdp_hold_last_plan_timeout": 4.0,
        "internvla_goal_depth_window": 5,
        "internvla_goal_depth_min": 0.25,
        "internvla_goal_depth_max": 6.0,
        "internvla_goal_update_min_dist": 0.35,
        "internvla_goal_filter_alpha": 0.35,
        "internvla_goal_confirm_samples": 2,
        "internvla_goal_min_stable_time": 0.6,
        "internvla_forward_step_m": 0.5,
        "internvla_turn_step_deg": 30.0,
        "internvla_action_timeout_s": 3.0,
        "nav_command_api_host": "127.0.0.1",
        "nav_command_api_port": 8892,
        "camera_api_host": "127.0.0.1",
        "camera_api_port": 8891,
        "camera_pitch_deg": 0.0,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


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
        self.reset_calls: list[dict[str, str]] = []
        self.results: list[System2Result] = []

    def reset(self, instruction: str, *, language: str = "auto") -> None:
        self.reset_calls.append({"instruction": str(instruction), "language": str(language)})

    def step_session(self, *, rgb, depth, stamp_s: float, session_id: str | None = None):  # noqa: ANN001
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


class _FakeSystem2AsyncPlanner:
    def __init__(self) -> None:
        self.outputs: list[AsyncSystem2Output | None] = []
        self.submitted: list[object] = []
        self.reset_calls = 0
        self.pending = False

    def reset_state(self) -> None:
        self.reset_calls += 1
        self.outputs.clear()
        self.submitted.clear()
        self.pending = False

    def submit(self, planner_input) -> None:  # noqa: ANN001
        self.submitted.append(planner_input)
        self.pending = True

    def has_pending_work(self) -> bool:
        return bool(self.pending)

    def consume_latest(self, last_seen_version: int):  # noqa: ANN001
        _ = last_seen_version
        if not self.outputs:
            return None
        self.pending = False
        return self.outputs.pop(0)


def _make_session(*, mode: str, args: Namespace | None = None) -> tuple[PlanningSession, _FakePlanner, _FakePlanner, _FakeNavDPClient, _FakeSystem2Client]:
    resolved_args = _args(planner_mode=mode) if args is None else args
    session = PlanningSession(resolved_args)
    session._intrinsic = np.asarray([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    session.navdp_client = _FakeNavDPClient()
    session.pointgoal_planner = _FakePlanner()
    session.nogoal_planner = _FakePlanner()
    session.system2_client = _FakeSystem2Client()
    return session, session.nogoal_planner, session.pointgoal_planner, session.navdp_client, session.system2_client


def _observation(frame_id: int) -> ExecutionObservation:
    return ExecutionObservation(
        frame_id=frame_id,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        sensor_meta={"rgb_source": "fake", "depth_source": "fake"},
        cam_pos=np.zeros(3, dtype=np.float32),
        cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        intrinsic=np.asarray([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )


def _system2_result(
    *,
    decision_mode: str,
    pixel_xy: tuple[float, float] | None = None,
    text: str = "",
    status: str = "goal",
    stamp_s: float = 1.0,
    action_sequence: tuple[str, ...] = (),
) -> System2Result:
    return System2Result(
        status=status,
        uv_norm=None if pixel_xy is None else np.asarray([0.5, 0.5], dtype=np.float32),
        text=text or decision_mode,
        latency_ms=12.0,
        stamp_s=float(stamp_s),
        pixel_xy=None if pixel_xy is None else np.asarray(pixel_xy, dtype=np.float32),
        decision_mode=decision_mode,
        action_sequence=action_sequence,
    )


def _step_engine(
    engine: PlannerRuntimeEngine,
    *,
    frame_id: int,
    robot_pos_world: np.ndarray | None = None,
    robot_yaw: float = 0.0,
) -> object:
    return engine.plan_with_observation(
        _observation(frame_id),
        action_command=_planner_managed_command(task_id="nav"),
        robot_pos_world=np.zeros(3, dtype=np.float32) if robot_pos_world is None else np.asarray(robot_pos_world, dtype=np.float32),
        robot_yaw=float(robot_yaw),
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_planning_session_default_navdp_client_factory_uses_supported_keyword_names(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_create_inprocess_navdp_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("runtime.planning_session.create_inprocess_navdp_client", _fake_create_inprocess_navdp_client)

    args = _args()
    session = PlanningSession(args)
    result = session._default_navdp_client_factory(np.eye(3, dtype=np.float32), args)

    assert result is not None
    assert captured["backend"] == "heuristic"
    assert captured["amp"] is False
    assert captured["tf32"] is False
    assert "backend_name" not in captured
    assert "use_amp" not in captured
    assert "allow_tf32" not in captured


def test_nav_task_start_resets_system2_and_navdp_exactly_once_per_instruction() -> None:
    args = _args(planner_mode="nav", s2_period_sec=999.0)
    session, _nogoal_planner, _pointgoal_planner, navdp_client, system2_client = _make_session(mode="nav", args=args)
    state = PlannerRuntimeState(mode="nav")
    engine = PlannerRuntimeEngine(args, transport=session, state=state)
    session.system2_planner = _FakeSystem2AsyncPlanner()
    system2_client.results = [_system2_result(decision_mode="wait", text="wait")]

    engine.start_nav_task("head to dock")
    _step_engine(engine, frame_id=1)
    _step_engine(engine, frame_id=2)

    assert system2_client.reset_calls == [{"instruction": "head to dock", "language": "auto"}]
    assert navdp_client.reset_calls == 1
    assert session.system2_planner.reset_calls == 1


def test_planner_runtime_engine_updates_nav_goal_from_system2_pixel_goal() -> None:
    args = _args(planner_mode="nav", internvla_goal_confirm_samples=2, s2_period_sec=0.0)
    session, _nogoal_planner, pointgoal_planner, navdp_client, system2_client = _make_session(mode="nav", args=args)
    state = PlannerRuntimeState(mode="nav")
    engine = PlannerRuntimeEngine(args, transport=session, state=state)

    system2_client.results = [
        _system2_result(decision_mode="pixel_goal", pixel_xy=(5.0, 4.0), text="pixel_goal"),
        _system2_result(decision_mode="pixel_goal", pixel_xy=(5.0, 4.0), text="pixel_goal", stamp_s=2.0),
    ]

    engine.start_nav_task("head to the dock")
    first = _step_engine(engine, frame_id=1)

    pointgoal_planner.outputs = [
        PlannerOutput(
            plan_version=0,
            source_frame_id=2,
            trajectory_world=np.asarray([[0.4, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32),
            latency_ms=3.5,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    pointgoal_planner.status = (1, 0, "", 3.5)
    second = _step_engine(engine, frame_id=2)

    assert system2_client.reset_calls == [{"instruction": "head to the dock", "language": "auto"}]
    assert navdp_client.reset_calls == 1
    assert first.goal_version == -1
    assert state.goal.system2_pixel_goal == [5, 4]
    assert second.goal_version == 0
    assert second.traj_version == 0
    assert second.trajectory_world.shape == (2, 3)
    assert len(pointgoal_planner.submitted) == 1
    submitted = pointgoal_planner.submitted[0]
    assert isinstance(submitted, PlannerInput)
    assert submitted.sensor_meta["robot_pose_xyz"] == [0.0, 0.0, 0.0]
    assert submitted.sensor_meta["robot_yaw_rad"] == 0.0
    assert state.navdp.last_committed_goal_version == 0
    assert state.navdp.last_committed_plan_version == 0


def test_planner_runtime_engine_uses_direct_action_overrides_and_queue_progress() -> None:
    args = _args(planner_mode="nav", s2_period_sec=999.0)
    session, _nogoal_planner, pointgoal_planner, _navdp_client, system2_client = _make_session(mode="nav", args=args)
    state = PlannerRuntimeState(mode="nav")
    engine = PlannerRuntimeEngine(args, transport=session, state=state)

    system2_client.results = [
        _system2_result(
            decision_mode="forward",
            text="forward then yaw left",
            action_sequence=("forward", "yaw_left"),
        )
    ]

    engine.start_nav_task("move toward the dock")
    first = _step_engine(engine, frame_id=1)
    second = _step_engine(engine, frame_id=2, robot_pos_world=np.asarray([0.5, 0.0, 0.0], dtype=np.float32))
    third = _step_engine(engine, frame_id=3, robot_pos_world=np.asarray([0.5, 0.0, 0.0], dtype=np.float32), robot_yaw=np.deg2rad(30.0))

    assert first.planner_control_mode == "forward"
    assert first.planner_control_queue == ("yaw_left",)
    assert second.planner_control_mode == "yaw_left"
    assert second.planner_control_queue == ()
    assert third.planner_control_mode == "wait"
    assert third.planner_control_progress == 1.0
    assert len(pointgoal_planner.submitted) == 0
    assert third.trajectory_world.shape == (0, 3)


def test_planner_runtime_engine_wait_preserves_active_goal() -> None:
    args = _args(planner_mode="nav", internvla_goal_confirm_samples=1)
    session, _nogoal_planner, _pointgoal_planner, _navdp_client, system2_client = _make_session(mode="nav", args=args)
    state = PlannerRuntimeState(mode="nav")
    engine = PlannerRuntimeEngine(args, transport=session, state=state)

    engine.start_nav_task("wait on the active goal")
    state.goal.target_world_xy = np.asarray([1.0, 0.0], dtype=np.float32)
    state.goal.target_pixel_xy = np.asarray([5.0, 4.0], dtype=np.float32)
    state.goal.goal_version = 3
    state.trajectory.planner_control_mode = "trajectory"
    system2_client.results = [_system2_result(decision_mode="wait", text="wait")]

    update = _step_engine(engine, frame_id=4)

    assert update.planner_control_mode == "trajectory"
    assert state.goal.target_world_xy is not None
    np.testing.assert_allclose(state.goal.target_world_xy, np.asarray([1.0, 0.0], dtype=np.float32))


def test_planner_runtime_engine_stop_clears_goal_and_zeroes_plan() -> None:
    args = _args(planner_mode="nav", s2_period_sec=0.0)
    session, _nogoal_planner, _pointgoal_planner, _navdp_client, system2_client = _make_session(mode="nav", args=args)
    state = PlannerRuntimeState(mode="nav")
    engine = PlannerRuntimeEngine(args, transport=session, state=state)

    engine.start_nav_task("stop the active goal")
    state.goal.target_world_xy = np.asarray([1.0, 0.0], dtype=np.float32)
    state.goal.goal_version = 1
    state.trajectory.trajectory_world = np.asarray([[0.3, 0.0, 0.0]], dtype=np.float32)
    state.trajectory.planner_control_mode = "trajectory"
    system2_client.results = [_system2_result(decision_mode="stop", text="stop", status="stop")]

    update = _step_engine(engine, frame_id=5)

    assert update.stop is True
    assert update.planner_control_mode == "stop"
    assert state.goal.target_world_xy is None
    assert state.trajectory.trajectory_world.shape == (0, 3)


def test_planner_runtime_engine_look_down_preserves_goal_and_marks_requery_hold() -> None:
    args = _args(planner_mode="nav", s2_period_sec=0.0)
    session, _nogoal_planner, _pointgoal_planner, _navdp_client, system2_client = _make_session(mode="nav", args=args)
    state = PlannerRuntimeState(mode="nav")
    engine = PlannerRuntimeEngine(args, transport=session, state=state)

    engine.start_nav_task("look down but preserve the goal")
    state.goal.target_world_xy = np.asarray([1.0, 0.0], dtype=np.float32)
    state.goal.goal_version = 2
    state.trajectory.planner_control_mode = "trajectory"
    system2_client.results = [_system2_result(decision_mode="look_down", text="look_down")]

    update = _step_engine(engine, frame_id=6)

    assert update.planner_control_mode == "trajectory"
    assert update.stale_hold_reason == "look_down_requery"
    assert state.goal.target_world_xy is not None


def test_planner_runtime_engine_stale_plan_hold_and_timeout_match_source_thresholds() -> None:
    args = _args(planner_mode="nav", navdp_plan_timeout=1.5, navdp_hold_last_plan_timeout=4.0)
    session, _nogoal_planner, _pointgoal_planner, _navdp_client, _system2_client = _make_session(mode="nav", args=args)
    state = PlannerRuntimeState(mode="nav")
    engine = PlannerRuntimeEngine(args, transport=session, state=state)

    state.goal.target_world_xy = np.asarray([2.0, 0.0], dtype=np.float32)
    state.goal.goal_version = 1
    state.trajectory.planner_control_mode = "trajectory"
    state.trajectory.trajectory_world = np.asarray([[0.4, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32)
    state.trajectory.last_plan_stamp_s = time.monotonic() - 2.0
    engine._refresh_stale_navdp_hold()  # noqa: SLF001

    assert state.trajectory.used_cached_traj is True
    assert state.trajectory.stale_hold_reason == "stale_hold"

    state.trajectory.last_plan_stamp_s = time.monotonic() - 5.0
    engine._refresh_stale_navdp_hold()  # noqa: SLF001

    assert state.trajectory.used_cached_traj is False
    assert state.trajectory.stale_hold_reason == "hold_timeout"
    assert state.trajectory.trajectory_world.shape == (0, 3)


def test_planner_runtime_engine_consumes_async_system2_results_without_direct_session_call() -> None:
    args = _args(planner_mode="nav", internvla_goal_confirm_samples=1, s2_period_sec=0.0)
    session, _nogoal_planner, pointgoal_planner, navdp_client, system2_client = _make_session(mode="nav", args=args)
    state = PlannerRuntimeState(mode="nav")
    engine = PlannerRuntimeEngine(args, transport=session, state=state)
    async_planner = _FakeSystem2AsyncPlanner()
    session.system2_planner = async_planner

    system2_client.step_session = lambda **kwargs: (_ for _ in ()).throw(AssertionError("direct System2 path should not be used"))  # type: ignore[method-assign]

    engine.start_nav_task("head to the async dock")
    pointgoal_planner.outputs = [
        PlannerOutput(
            plan_version=0,
            source_frame_id=4,
            trajectory_world=np.asarray([[0.25, 0.0, 0.0], [0.55, 0.1, 0.0]], dtype=np.float32),
            latency_ms=2.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    pointgoal_planner.status = (1, 0, "", 2.0)
    async_planner.outputs = [
        AsyncSystem2Output(
            result_version=0,
            source_frame_id=4,
            result=_system2_result(decision_mode="pixel_goal", pixel_xy=(5.0, 4.0), text="pixel_goal"),
        )
    ]
    update = _step_engine(engine, frame_id=4)

    assert system2_client.reset_calls == [{"instruction": "head to the async dock", "language": "auto"}]
    assert async_planner.reset_calls == 1
    assert navdp_client.reset_calls == 1
    assert state.goal.system2_result_version == 0
    assert state.goal.system2_pixel_goal == [5, 4]
    assert update.goal_version == 0
    assert update.traj_version == 0
    assert len(pointgoal_planner.submitted) == 1
