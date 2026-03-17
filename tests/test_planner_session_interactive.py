from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from control.async_planners import DualPlannerOutput, PlannerOutput
from ipc.messages import ActionCommand
import runtime.planning_session as planning_session_module
from runtime.planning_session import PlanningSession


def _args(*, planner_mode: str = "interactive", instruction: str = "") -> Namespace:
    return Namespace(
        planner_mode=planner_mode,
        server_url="http://127.0.0.1:8888",
        dual_server_url="http://127.0.0.1:8890",
        instruction=instruction,
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
    )


def _planner_managed_command(task_id: str = "interactive") -> ActionCommand:
    return ActionCommand(
        action_type="LOCAL_SEARCH",
        task_id=task_id,
        metadata={"planner_managed": True},
    )


def _planner_control(mode: str = "trajectory", *, yaw_delta_rad: float | None = None, reason: str = "") -> dict[str, object]:
    return {
        "mode": mode,
        "yaw_delta_rad": yaw_delta_rad,
        "reason": reason,
    }


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


def _step(session: PlanningSession, frame_id: int):
    return session.plan_with_observation(
        _observation(session, frame_id),
        action_command=_planner_managed_command(),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


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


class _FakePopen:
    def __init__(self, returncode: int | None = None) -> None:
        self._returncode = returncode

    def poll(self) -> int | None:
        return self._returncode


def _make_session() -> tuple[PlanningSession, _FakePlanner, _FakePlanner, _FakeNavDPClient, _FakeDualClient]:
    session = PlanningSession(_args())
    session._intrinsic = np.eye(3, dtype=np.float32)
    nogoal_planner = _FakePlanner()
    dual_planner = _FakePlanner()
    navdp_client = _FakeNavDPClient()
    dual_client = _FakeDualClient()
    session.navdp_client = navdp_client
    session.nogoal_planner = nogoal_planner
    session.dual_planner = dual_planner
    session._dual_client = dual_client
    assert session._activate_roaming("test startup") is True
    return session, nogoal_planner, dual_planner, navdp_client, dual_client


def test_ensure_remote_service_ready_autostarts_local_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    checks = {"count": 0}
    launches: list[Path] = []

    def _fake_check(base_url: str, *, timeout_sec: float, service_name: str, context: str) -> None:
        assert base_url == "http://127.0.0.1:8890"
        assert timeout_sec == 1.0
        assert service_name == "dual server"
        assert context == "interactive task (stdin)"
        checks["count"] += 1
        if checks["count"] == 1:
            raise RuntimeError("interactive task (stdin): dual server is unavailable at http://127.0.0.1:8890")

    def _fake_resolve(script_name: str) -> Path:
        assert script_name == "run_vlm_dual_server.ps1"
        return Path("run_vlm_dual_server.ps1")

    def _fake_start(script_path: Path) -> _FakePopen:
        launches.append(script_path)
        return _FakePopen()

    monkeypatch.setattr(planning_session_module, "_check_remote_service", _fake_check)
    monkeypatch.setattr(planning_session_module, "_resolve_local_launcher_script", _fake_resolve)
    monkeypatch.setattr(planning_session_module, "_start_local_launcher", _fake_start)
    monkeypatch.setattr(planning_session_module.time, "sleep", lambda _: None)

    launcher_processes: dict[str, _FakePopen] = {}
    planning_session_module._ensure_remote_service_ready(
        "http://127.0.0.1:8890",
        timeout_sec=1.0,
        service_name="dual server",
        context="interactive task (stdin)",
        launcher_script_name="run_vlm_dual_server.ps1",
        launcher_processes=launcher_processes,
        startup_timeout_sec=1.0,
    )

    assert checks["count"] == 2
    assert launches == [Path("run_vlm_dual_server.ps1")]
    assert "http://127.0.0.1:8890" in launcher_processes


def test_ensure_remote_service_ready_does_not_autostart_remote_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planning_session_module,
        "_check_remote_service",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("remote dual server unavailable")),
    )

    with pytest.raises(RuntimeError, match="remote dual server unavailable"):
        planning_session_module._ensure_remote_service_ready(
            "http://10.0.0.5:8890",
            timeout_sec=1.0,
            service_name="dual server",
            context="interactive task (stdin)",
            launcher_script_name="run_vlm_dual_server.ps1",
            launcher_processes={},
            startup_timeout_sec=1.0,
        )


def test_interactive_roaming_uses_nogoal_planner() -> None:
    session, nogoal_planner, _dual_planner, navdp_client, _dual_client = _make_session()
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

    update = _step(session, 1)

    assert navdp_client.reset_calls == 1
    assert len(nogoal_planner.submitted) == 1
    assert update.interactive_phase == "roaming"
    assert update.trajectory_world.shape == (2, 3)


def test_interactive_command_switches_to_task_mode_and_resets_dual() -> None:
    session, _nogoal_planner, dual_planner, _navdp_client, dual_client = _make_session()
    dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=2,
            trajectory_world=np.asarray([[0.3, 0.0, 0.0], [0.5, 0.1, 0.0]], dtype=np.float32),
            pixel_goal=None,
            stop=False,
            goal_version=3,
            traj_version=4,
            stale_sec=0.2,
            used_cached_traj=False,
            planner_control=_planner_control(),
            debug={},
            latency_ms=4.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    dual_planner.status = (1, 0, "", 4.0)
    command_id = session.submit_interactive_instruction("go to the loading dock")

    update = _step(session, 2)

    assert command_id == 1
    assert dual_client.instructions == ["go to the loading dock"]
    assert update.interactive_phase == "task_active"
    assert update.interactive_command_id == 1
    assert update.goal_version == 3
    assert update.traj_version == 4


def test_interactive_completion_returns_to_roaming() -> None:
    session, nogoal_planner, dual_planner, navdp_client, dual_client = _make_session()
    dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=1,
            trajectory_world=np.asarray([[0.3, 0.0, 0.0], [0.5, 0.1, 0.0]], dtype=np.float32),
            pixel_goal=None,
            stop=False,
            goal_version=1,
            traj_version=2,
            stale_sec=0.1,
            used_cached_traj=False,
            planner_control=_planner_control(),
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
            pixel_goal=None,
            stop=True,
            goal_version=2,
            traj_version=3,
            stale_sec=0.0,
            used_cached_traj=False,
            planner_control=_planner_control("stop", reason="STOP"),
            debug={},
            latency_ms=4.0,
            successful_calls=2,
            failed_calls=0,
            last_error="",
        ),
    ]
    dual_planner.status = (2, 0, "", 4.0)
    session.submit_interactive_instruction("inspect the pallet")

    first_update = _step(session, 1)
    second_update = _step(session, 2)

    assert first_update.interactive_phase == "task_active"
    assert second_update.interactive_phase == "roaming"
    assert second_update.interactive_command_id == -1
    assert second_update.stop is True
    assert dual_client.instructions == ["inspect the pallet"]
    assert navdp_client.reset_calls == 2

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

    roaming_update = _step(session, 3)

    assert roaming_update.interactive_phase == "roaming"
    assert roaming_update.trajectory_world.shape == (2, 3)


def test_interactive_new_instruction_preempts_active_task() -> None:
    session, _nogoal_planner, dual_planner, _navdp_client, dual_client = _make_session()
    dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=1,
            trajectory_world=np.asarray([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32),
            pixel_goal=None,
            stop=False,
            goal_version=1,
            traj_version=1,
            stale_sec=0.1,
            used_cached_traj=False,
            planner_control=_planner_control(),
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
            pixel_goal=None,
            stop=False,
            goal_version=2,
            traj_version=2,
            stale_sec=0.1,
            used_cached_traj=False,
            planner_control=_planner_control(),
            debug={},
            latency_ms=4.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        ),
    ]
    dual_planner.status = (1, 0, "", 4.0)

    first_id = session.submit_interactive_instruction("go forward")
    first_update = _step(session, 1)
    second_id = session.submit_interactive_instruction("turn left and move")
    second_update = _step(session, 2)

    assert first_id == 1
    assert second_id == 2
    assert first_update.interactive_command_id == 1
    assert second_update.interactive_phase == "task_active"
    assert second_update.interactive_command_id == 2
    assert dual_client.instructions == ["go forward", "turn left and move"]


def test_interactive_cancel_returns_to_roaming() -> None:
    session, _nogoal_planner, dual_planner, navdp_client, _dual_client = _make_session()
    dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=1,
            trajectory_world=np.asarray([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32),
            pixel_goal=None,
            stop=False,
            goal_version=1,
            traj_version=1,
            stale_sec=0.1,
            used_cached_traj=False,
            planner_control=_planner_control(),
            debug={},
            latency_ms=4.0,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    dual_planner.status = (1, 0, "", 4.0)
    session.submit_interactive_instruction("go forward")
    task_update = _step(session, 1)

    assert task_update.interactive_phase == "task_active"
    assert session.cancel_interactive_task() is True

    cancel_update = _step(session, 2)

    assert cancel_update.interactive_phase == "roaming"
    assert cancel_update.interactive_command_id == -1
    assert navdp_client.reset_calls == 2


def test_dual_mode_uses_dual_server_for_trajectory_generation() -> None:
    session = PlanningSession(_args(planner_mode="dual", instruction="head to the dock"))
    session._intrinsic = np.eye(3, dtype=np.float32)
    session.navdp_client = _FakeNavDPClient()
    session.pointgoal_planner = _FakePlanner()
    session.nogoal_planner = _FakePlanner()
    session.dual_planner = _FakePlanner()
    dual_client = _FakeDualClient()
    session._dual_client = dual_client
    session.dual_planner.outputs = [
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
            planner_control=_planner_control(),
            debug={},
            latency_ms=3.5,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    session.dual_planner.status = (1, 0, "", 3.5)

    session.start_dual_task("head to the dock")
    update = session.plan_with_observation(
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
    assert session.viewer_overlay_state()["system2_pixel_goal"] == [240, 180]


def test_dual_mode_exposes_planner_managed_yaw_control() -> None:
    session = PlanningSession(_args(planner_mode="dual", instruction="turn toward the dock"))
    session._intrinsic = np.eye(3, dtype=np.float32)
    session.navdp_client = _FakeNavDPClient()
    session.pointgoal_planner = _FakePlanner()
    session.nogoal_planner = _FakePlanner()
    session.dual_planner = _FakePlanner()
    dual_client = _FakeDualClient()
    session._dual_client = dual_client
    session.dual_planner.outputs = [
        DualPlannerOutput(
            plan_version=0,
            source_frame_id=1,
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            pixel_goal=None,
            stop=False,
            goal_version=6,
            traj_version=7,
            stale_sec=-1.0,
            used_cached_traj=False,
            planner_control=_planner_control("yaw_delta", yaw_delta_rad=float(np.pi / 6.0), reason="←"),
            debug={},
            latency_ms=3.5,
            successful_calls=1,
            failed_calls=0,
            last_error="",
        )
    ]
    session.dual_planner.status = (1, 0, "", 3.5)

    session.start_dual_task("turn toward the dock")
    update = session.plan_with_observation(
        _observation(session, 1),
        action_command=_planner_managed_command(task_id="dual"),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert update.trajectory_world.shape == (0, 3)
    assert update.planner_control_mode == "yaw_delta"
    assert update.planner_yaw_delta_rad == float(np.pi / 6.0)
