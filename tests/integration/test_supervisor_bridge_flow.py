from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from systems.transport.messages import ActionCommand
from runtime.aura_runtime import AuraRuntimeCommandSource
from runtime.planning_session import ExecutionObservation, PlannerStats, TrajectoryUpdate


class _FakeSimulationApp:
    def update(self) -> None:
        return None


class _FakeController:
    def get_base_state(self):
        return SimpleNamespace(
            position_w=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            lin_vel_w=np.zeros(3, dtype=np.float32),
            ang_vel_w=np.zeros(3, dtype=np.float32),
        )


class _FakeBridge:
    def __init__(self) -> None:
        self.notices = []
        self.statuses = []
        self.health = []

    def publish_notice(self, notice) -> None:  # noqa: ANN001
        self.notices.append(notice)

    def publish_status(self, status) -> None:  # noqa: ANN001
        self.statuses.append(status)

    def publish_health(self, payload) -> None:  # noqa: ANN001
        self.health.append(payload)

    def publish_capability(self, payload) -> None:  # noqa: ANN001
        _ = payload

    def drain_task_requests(self):
        return []

    def drain_runtime_controls(self):
        return []


class _FakeSupervisor:
    def __init__(self) -> None:
        self.bridge = _FakeBridge()
        self.memory_service = SimpleNamespace(scratchpad=SimpleNamespace(task_id="interactive"))
        self.perception_pipeline = SimpleNamespace(detector=SimpleNamespace(runtime_report=None, info=SimpleNamespace(backend_name="stub")))


class _FakePlanningSession:
    def __init__(self) -> None:
        self.last_action_type = ""
        self._plan_version = -1
        self.started_instructions: list[str] = []
        self.health_checks: list[str] = []
        self.submitted_instructions: list[str] = []
        self._viewer_trajectory_world = [
            [0.0, 0.0, 2.0],
            [0.4, 0.0, 2.0],
            [0.8, 0.0, 2.0],
        ]

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        _ = simulation_app, stage

    def capture_observation(self, frame_id: int, env=None):  # noqa: ANN001
        _ = env
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        rgb[24:72, 28:68, 0] = 255
        depth = np.full((96, 96), 1.5, dtype=np.float32)
        intrinsic = np.asarray([[96.0, 0.0, 48.0], [0.0, 96.0, 48.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return ExecutionObservation(
            frame_id=int(frame_id),
            rgb=rgb,
            depth=depth,
            sensor_meta={"target_class_hint": "apple", "color_hint": "red", "room_id": "kitchen"},
            cam_pos=np.asarray([0.0, 0.0, 1.2], dtype=np.float32),
            cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            intrinsic=intrinsic,
        )

    def plan_with_observation(self, observation, *, action_command, robot_pos_world, robot_yaw, robot_quat_wxyz):  # noqa: ANN001
        _ = observation, robot_pos_world, robot_yaw, robot_quat_wxyz
        self.last_action_type = action_command.action_type if action_command is not None else ""
        self._plan_version += 1
        return TrajectoryUpdate(
            trajectory_world=np.asarray([[0.3, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float32),
            plan_version=self._plan_version,
            stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=2.0, last_plan_step=int(observation.frame_id)),
            source_frame_id=int(observation.frame_id),
            action_command=action_command,
            stop=False,
            interactive_phase="task_active" if self.submitted_instructions else None,
            interactive_command_id=len(self.submitted_instructions) if self.submitted_instructions else -1,
            interactive_instruction=self.submitted_instructions[-1] if self.submitted_instructions else "",
        )

    def ensure_navdp_service_ready(self, *, context: str) -> None:
        self.health_checks.append(f"navdp:{context}")

    def ensure_system2_service_ready(self, *, context: str) -> None:
        self.health_checks.append(f"system2:{context}")

    def start_nav_task(self, instruction: str) -> None:
        self.started_instructions.append(str(instruction))

    def submit_interactive_instruction(self, instruction: str) -> int:
        self.submitted_instructions.append(str(instruction))
        return len(self.submitted_instructions)

    def cancel_interactive_task(self) -> bool:
        return True

    def viewer_overlay_state(self) -> dict[str, object]:
        return {
            "trajectory_world": list(self._viewer_trajectory_world),
            "plan_version": 4,
            "goal_version": 2,
            "traj_version": 3,
        }

    def shutdown(self) -> None:
        return None

    def active_memory_instruction(self) -> str:
        if self.submitted_instructions:
            return self.submitted_instructions[-1]
        return self.started_instructions[-1] if self.started_instructions else ""


class _FakeServer:
    def __init__(self, args: Namespace, planning_session: _FakePlanningSession) -> None:
        self.args = args
        self.planning_session = planning_session

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        _ = simulation_app, stage

    def bootstrap(self):
        mode = str(getattr(self.args, "planner_mode", "")).strip().lower()
        if mode == "interactive":
            self.planning_session.ensure_navdp_service_ready(context="interactive startup")
        elif mode == "nav":
            self.planning_session.ensure_navdp_service_ready(context="nav startup")
            self.planning_session.ensure_system2_service_ready(context="nav startup")
            self.planning_session.start_nav_task(str(getattr(self.args, "instruction", "")))
        return []

    def tick(self, *, frame_event, task_events, runtime_status, robot_pos_world, robot_lin_vel_world, robot_ang_vel_world, robot_yaw, robot_quat_wxyz):  # noqa: ANN001
        _ = task_events, runtime_status, robot_lin_vel_world, robot_ang_vel_world
        action_command = ActionCommand(action_type="LOCAL_SEARCH", task_id=str(frame_event.metadata.task_id), metadata={"planner_managed": True})
        update = self.planning_session.plan_with_observation(
            frame_event.observation,
            action_command=action_command,
            robot_pos_world=robot_pos_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
        )
        return SimpleNamespace(
            notices=[],
            trajectory_update=update,
            evaluation=SimpleNamespace(goal_distance_m=1.0, yaw_error_rad=0.0, reached_goal=False),
            command_vector=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
            status=None,
            action_command=action_command,
            viewer_overlay=self.planning_session.viewer_overlay_state(),
        )

    def snapshot(self):
        return None

    def shutdown(self) -> None:
        return None


def _args() -> Namespace:
    return Namespace(
        planner_mode="nav",
        instruction="go to the loading dock",
        spawn_demo_object=False,
        goal_x=None,
        goal_y=None,
        goal_tolerance_m=0.4,
        startup_updates=0,
        cmd_max_vx=0.5,
        cmd_max_vy=0.3,
        cmd_max_wz=0.8,
        lookahead_distance_m=0.6,
        heading_slowdown_rad=0.6,
        traj_stale_timeout_sec=1.5,
        cmd_accel_limit=1.0,
        cmd_yaw_accel_limit=1.5,
        log_interval=1,
        interactive_idle_log_interval=120,
        memory_db_path="state/memory/memory.sqlite",
        detector_model_path="artifacts/models/__missing__.engine",
        detector_device="",
        timeout_sec=5.0,
        launch_mode="headless",
        resolved_launch_mode="headless",
        headless=True,
        viewer_publish=False,
        native_viewer="off",
        viewer_control_endpoint="tcp://127.0.0.1:5580",
        viewer_telemetry_endpoint="tcp://127.0.0.1:5581",
        viewer_shm_name="g1_view_frames",
        viewer_shm_slot_size=2 * 1024 * 1024,
        viewer_shm_capacity=4,
    )


def _install_fakes(monkeypatch: pytest.MonkeyPatch, command_source: AuraRuntimeCommandSource, planning_session: _FakePlanningSession) -> None:
    fake_supervisor = _FakeSupervisor()

    def fake_runtime_bridge(self) -> None:  # noqa: ANN001
        self._runtime_io = SimpleNamespace(bus=SimpleNamespace(), shm_ring=None, close=lambda unlink_shm=True: None)
        self._supervisor = fake_supervisor

    def fake_control_server(self) -> None:  # noqa: ANN001
        if self._server is None:
            self._server = _FakeServer(self.args, planning_session)

    monkeypatch.setattr(AuraRuntimeCommandSource, "_ensure_runtime_bridge", fake_runtime_bridge)
    monkeypatch.setattr(AuraRuntimeCommandSource, "_ensure_control_server", fake_control_server)


def test_nav_mode_bootstraps_direct_system2_and_navdp_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    planning_session = _FakePlanningSession()
    command_source = AuraRuntimeCommandSource(_args(), planning_session=planning_session)
    _install_fakes(monkeypatch, command_source, planning_session)

    command_source.initialize(_FakeSimulationApp(), stage=None, controller=_FakeController())
    command_source.update(1)

    assert planning_session.started_instructions == ["go to the loading dock"]
    assert planning_session.health_checks == ["navdp:nav startup", "system2:nav startup"]
    assert planning_session.last_action_type == "LOCAL_SEARCH"
    assert np.allclose(command_source.command(), np.asarray([0.1, 0.0, 0.0], dtype=np.float32))


def test_interactive_mode_keeps_roaming_bootstrap_and_accepts_runtime_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    planning_session = _FakePlanningSession()
    args = _args()
    args.planner_mode = "interactive"
    args.instruction = ""
    command_source = AuraRuntimeCommandSource(args, planning_session=planning_session)
    _install_fakes(monkeypatch, command_source, planning_session)

    command_source.initialize(_FakeSimulationApp(), stage=None, controller=_FakeController())
    command_source.planning_session.submit_interactive_instruction("go to the loading dock")
    command_source.update(1)

    assert planning_session.health_checks == ["navdp:interactive startup"]
    assert planning_session.submitted_instructions == ["go to the loading dock"]
    assert planning_session.last_action_type == "LOCAL_SEARCH"
    assert np.allclose(command_source.command(), np.asarray([0.1, 0.0, 0.0], dtype=np.float32))
