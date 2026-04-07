from __future__ import annotations

import os
import time
import traceback
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from apps.runtime_common import RuntimeIo, build_runtime_io
from ipc.inproc_bus import InprocBus
from ipc.messages import (
    ActionCommand,
    ActionStatus,
    RuntimeControlRequest,
    TaskRequest,
)
from ipc.zmq_bus import ZmqBus
from runtime_pipeline.bootstrap import RuntimeBootstrapper
from runtime_pipeline.ingestion import FrameIngestor
from runtime_pipeline.publication import RuntimePublisher

from .aura_runtime_args import (
    apply_demo_defaults,
    apply_launch_mode_defaults,
    build_arg_parser,
    build_launch_config,
    resolve_launch_mode,
    validate_args,
)
from .planning_session import PlanningSession
from .subgoal_executor import CommandEvaluation, SubgoalExecutor

if TYPE_CHECKING:
    from server.main_control_server import MainControlServer
    from .supervisor import Supervisor


def _trace_runtime_event(message: str) -> None:
    path = str(os.environ.get("AURA_RUNTIME_TRACE_PATH", "")).strip()
    if path == "":
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        return


def _trace_runtime_exception(exc: BaseException) -> None:
    _trace_runtime_event(f"unhandled exception: {type(exc).__name__}: {exc}")
    tb = traceback.format_exc()
    for line in tb.splitlines():
        _trace_runtime_event(f"traceback: {line}")


class AuraRuntimeCommandSource:
    def __init__(
        self,
        args,
        *,
        supervisor=None,
        planning_session: PlanningSession | None = None,
    ) -> None:
        self.args = args
        self.quit_requested = False
        self.exit_code = 0
        self.shutdown_reason = ""
        self._launch_mode = str(getattr(args, "resolved_launch_mode", resolve_launch_mode(args))).strip().lower()
        self._viewer_publish = bool(getattr(args, "viewer_publish", self._launch_mode == "g1_view"))
        self._native_viewer = str(getattr(args, "native_viewer", "off")).strip().lower() or "off"
        self.requires_render = self._launch_mode in {"g1_view", "headless"}

        self._controller = None
        self._planning_session = planning_session or PlanningSession(args)
        self._executor = SubgoalExecutor(args)
        self._supervisor_config_kwargs = {
            "memory_db_path": str(getattr(args, "memory_db_path", "state/memory/memory.sqlite")),
            "detector_model_path": str(getattr(args, "detector_model_path", "")),
            "detector_device": str(getattr(args, "detector_device", "")),
            "memory_store": bool(getattr(args, "memory_store", True)),
            "skip_detection": bool(getattr(args, "skip_detection", False)),
        }
        self._supervisor = supervisor
        self._runtime_io: RuntimeIo | None = None
        self._command = np.zeros(3, dtype=np.float32)
        self._active_command: ActionCommand | None = None
        self._pending_status: ActionStatus | None = None
        self._manual_command: ActionCommand | None = None
        self._last_robot_pose_xyz = (0.0, 0.0, 0.0)
        self._pending_exit_code: int | None = None
        self._pending_exit_frames = 0
        self._pending_exit_reason = ""
        self._last_viewer_overlay: dict[str, object] = {}
        self._last_runtime_snapshot_frame = -1
        self._server = None
        self._runtime_bootstrapper = None
        self._frame_ingestor = None
        self._runtime_publisher = None

    @property
    def supervisor(self):
        if self._supervisor is None:
            from .supervisor import Supervisor

            self._supervisor = Supervisor(config=self._create_supervisor_config())
        return self._supervisor

    def _create_supervisor_config(self):
        from .supervisor import SupervisorConfig

        return SupervisorConfig(**self._supervisor_config_kwargs)

    @property
    def planning_session(self) -> PlanningSession:
        return self._planning_session

    def initialize(self, simulation_app, stage, controller) -> None:
        self._bootstrap_service().initialize(simulation_app, stage, controller)

    def update(self, frame_idx: int) -> None:
        self._seed_planner_overlay()
        capture = self._frame_ingestion_service().capture(frame_idx)
        if capture.batch is not None:
            capture.batch.frame_header.metadata["active_command_overlay"] = self._command_overlay_metadata()
        self._last_robot_pose_xyz = capture.robot_pose_xyz
        self._ensure_control_server()
        assert self._server is not None
        incoming_status = self._pending_status
        tick_result = self._server.tick(
            frame_event=capture.frame_event,
            task_events=capture.runtime_events,
            runtime_status=incoming_status,
            robot_pos_world=np.asarray(capture.base_state.position_w, dtype=np.float32),
            robot_lin_vel_world=np.asarray(capture.base_state.lin_vel_w, dtype=np.float32),
            robot_ang_vel_world=np.asarray(capture.base_state.ang_vel_w, dtype=np.float32),
            robot_yaw=capture.robot_yaw_rad,
            robot_quat_wxyz=capture.robot_quat_wxyz,
        )
        update = tick_result.trajectory_update
        evaluation = tick_result.evaluation
        self._command = tick_result.command_vector.copy()
        self._pending_status = tick_result.status
        self._active_command = tick_result.action_command
        self._last_viewer_overlay = dict(tick_result.viewer_overlay)
        self._publisher_service().publish_tick(
            frame_idx=frame_idx,
            notices=tick_result.notices,
            status=self._pending_status,
        )

        if self._pending_exit_code is not None:
            if self._pending_exit_frames <= 0:
                reason = self._pending_exit_reason if self._pending_exit_reason != "" else "quit requested"
                print(f"{self._log_prefix()} shutdown reason: {reason}")
                self.quit_requested = True
                self.exit_code = int(self._pending_exit_code)
                self.shutdown_reason = reason
            else:
                self._pending_exit_frames -= 1

        log_interval = max(int(self.args.log_interval), 1)
        if frame_idx % log_interval == 0:
            self._log_step(frame_idx, update, tick_result.action_command, evaluation)

    def command(self) -> np.ndarray:
        return self._command.copy()

    def shutdown(self) -> None:
        self._publish_notice(level="info", notice="aura runtime shutdown", details={"reason": self.shutdown_reason})
        if getattr(self, "_server", None) is not None:
            self._server.shutdown()
        else:
            self._executor.shutdown()
        if self._runtime_io is not None:
            self._runtime_io.close(unlink_shm=True)
            self._runtime_io = None

    def _bootstrap_mode(self) -> None:
        assert self._server is not None
        for notice in self._server.bootstrap():
            self._publish_notice(
                level=str(getattr(notice, "level", "info")),
                notice=str(getattr(notice, "notice", "")),
                details=getattr(notice, "details", {}),
            )

    def _bootstrap_service(self) -> RuntimeBootstrapper:
        bootstrapper = getattr(self, "_runtime_bootstrapper", None)
        if bootstrapper is None:
            bootstrapper = RuntimeBootstrapper(
                startup_updates=max(int(getattr(self.args, "startup_updates", 0)), 0),
                controller_setter=self._set_controller,
                ensure_runtime_bridge=self._ensure_runtime_bridge,
                ensure_control_server=self._ensure_control_server,
                initialize_server=self._initialize_control_server,
                bootstrap_mode=self._bootstrap_mode,
                seed_planner_overlay=self._seed_planner_overlay,
                publish_detector_capability=self._publish_detector_capability,
                publish_ready_notice=self._publish_runtime_ready_notice,
            )
            self._runtime_bootstrapper = bootstrapper
        return bootstrapper

    def _frame_ingestion_service(self) -> FrameIngestor:
        ingestor = getattr(self, "_frame_ingestor", None)
        if ingestor is None:
            ingestor = FrameIngestor(
                controller_provider=lambda: self._controller,
                planning_session_provider=lambda: self.planning_session,
                supervisor_provider=lambda: self.supervisor,
                runtime_events_provider=self._drain_external_runtime_requests,
                planner_overlay_provider=lambda: dict(self._last_viewer_overlay),
                viewer_publish=bool(self._viewer_publish),
                timeout_sec=float(getattr(self.args, "timeout_sec", 5.0)),
            )
            self._frame_ingestor = ingestor
        return ingestor

    def _publisher_service(self) -> RuntimePublisher:
        publisher = getattr(self, "_runtime_publisher", None)
        if publisher is None:
            publisher = RuntimePublisher(
                runtime_io_provider=lambda: self._runtime_io,
                bridge_provider=self._bridge,
                detector_report_provider=self._detector_runtime_report,
                detector_backend_provider=self._detector_backend_name,
                snapshot_provider=self._server_snapshot,
                snapshot_interval_provider=self._snapshot_interval,
                last_snapshot_frame_getter=lambda: int(self._last_runtime_snapshot_frame),
                last_snapshot_frame_setter=self._set_last_runtime_snapshot_frame,
            )
            self._runtime_publisher = publisher
        return publisher

    def _set_controller(self, controller) -> None:  # noqa: ANN001
        self._controller = controller

    def _initialize_control_server(self, simulation_app, stage) -> None:  # noqa: ANN001
        assert self._server is not None
        self._server.initialize(simulation_app, stage)

    def _publish_runtime_ready_notice(self) -> None:
        self._publish_notice(
            level="info",
            notice="aura runtime ready",
            details={
                "launchMode": self._launch_mode,
                "viewerPublish": bool(self._viewer_publish),
                "nativeViewer": self._native_viewer,
            },
        )

    def _bridge(self):  # noqa: ANN201
        if self._supervisor is None:
            return None
        return self._supervisor.bridge

    def _detector_runtime_report(self):  # noqa: ANN201
        if self._supervisor is None:
            return None
        return self._supervisor.perception_pipeline.detector.runtime_report

    def _detector_backend_name(self) -> str:
        if self._supervisor is None:
            return ""
        return str(self._supervisor.perception_pipeline.detector.info.backend_name)

    def _server_snapshot(self):  # noqa: ANN201
        return None if self._server is None else self._server.snapshot()

    def _snapshot_interval(self) -> int:
        return max(int(getattr(self.args, "log_interval", 30)), 1)

    def _set_last_runtime_snapshot_frame(self, frame_idx: int) -> None:
        self._last_runtime_snapshot_frame = int(frame_idx)

    def _seed_planner_overlay(self) -> None:
        if self._last_viewer_overlay:
            return
        getter = getattr(self.planning_session, "viewer_overlay_state", None)
        if not callable(getter):
            return
        state = getter()
        if isinstance(state, dict):
            self._last_viewer_overlay = dict(state)

    def _interactive_input_loop(self) -> None:
        return None

    def _command_overlay_metadata(self) -> dict[str, object]:
        command = self._active_command or self._manual_command
        if command is None:
            return {}
        metadata = dict(command.metadata)
        overlay: dict[str, object] = {"action_type": str(command.action_type)}
        for key in (
            "target_mode",
            "target_class",
            "target_track_id",
            "pose_source",
            "raw_target_pose_xyz",
            "filtered_target_pose_xyz",
            "nav_goal_pose_xyz",
            "approach_yaw_rad",
            "track_age_sec",
            "depth_m",
        ):
            if key in metadata:
                overlay[key] = metadata[key]
        if command.target_track_id != "" and "target_track_id" not in overlay:
            overlay["target_track_id"] = str(command.target_track_id)
        if command.target_pose_xyz is not None and "nav_goal_pose_xyz" not in overlay:
            overlay["nav_goal_pose_xyz"] = list(command.target_pose_xyz)
        return overlay

    def _resolve_action_command(self, *, robot_pose: tuple[float, float, float]) -> ActionCommand | None:
        if self._manual_command is not None:
            self._active_command = self._manual_command
            return self._manual_command
        command = self.supervisor.step(
            now=time.time(),
            robot_pose=robot_pose,
            action_status=self._pending_status,
            publish=False,
        )
        self._pending_status = None
        self._active_command = command
        return command

    def _arm_exit(self, exit_code: int, reason: str) -> None:
        if self._pending_exit_code is None:
            self._pending_exit_code = int(exit_code)
            self._pending_exit_reason = str(reason)
            self._pending_exit_frames = 1

    def _ensure_runtime_bridge(self) -> None:
        if self._runtime_io is not None:
            return
        if self._viewer_publish:
            self._runtime_io = build_runtime_io(
                bus_kind="zmq",
                endpoint=str(getattr(self.args, "viewer_control_endpoint", "")),
                bind=True,
                shm_name=str(getattr(self.args, "viewer_shm_name", "g1_view_frames")),
                shm_slot_size=int(getattr(self.args, "viewer_shm_slot_size", 8 * 1024 * 1024)),
                shm_capacity=int(getattr(self.args, "viewer_shm_capacity", 8)),
                create_shm=True,
                role="bridge",
                control_endpoint=str(getattr(self.args, "viewer_control_endpoint", "")),
                telemetry_endpoint=str(getattr(self.args, "viewer_telemetry_endpoint", "")),
            )
        else:
            try:
                bus = ZmqBus(
                    control_endpoint=str(getattr(self.args, "viewer_control_endpoint", "")),
                    telemetry_endpoint=str(getattr(self.args, "viewer_telemetry_endpoint", "")),
                    role="bridge",
                )
            except Exception:  # noqa: BLE001
                bus = InprocBus()
            self._runtime_io = RuntimeIo(
                bus=bus,
                shm_ring=None,
            )
        from .supervisor import Supervisor

        self._supervisor = Supervisor(
            bus=self._runtime_io.bus,
            shm_ring=self._runtime_io.shm_ring,
            config=self._create_supervisor_config(),
        )
        print(
            "[AURA_RUNTIME] runtime bridge ready "
            f"control={getattr(self.args, 'viewer_control_endpoint', '')} "
            f"telemetry={getattr(self.args, 'viewer_telemetry_endpoint', '')} "
            f"shm={getattr(self.args, 'viewer_shm_name', '') if self._viewer_publish else 'disabled'} "
            f"viewer_publish={self._viewer_publish}"
        )

    def _ensure_control_server(self) -> None:
        if self._server is not None:
            return
        from server.main_control_server import MainControlServer

        self._server = MainControlServer(
            self.args,
            supervisor=self.supervisor,
            planning_session=self._planning_session,
            executor=self._executor,
        )

    def _log_prefix(self) -> str:
        return "[AURA_RUNTIME]"

    def _submit_task_request(self, text: str, *, source: str, task_id: str = "") -> int:
        assert self._server is not None
        notices = self._server.submit_task_request(TaskRequest(command_text=text, task_id=str(task_id or "task")))
        if notices:
            notice = notices[-1]
            self._publish_notice(level=notice.level, notice=notice.notice, details=notice.details)
        return -1

    def _set_idle(self, *, source: str) -> bool:
        assert self._server is not None
        cancelled, notice = self._server.set_idle(source=source)
        if notice is not None:
            self._publish_notice(level=notice.level, notice=notice.notice, details=notice.details)
        return cancelled

    def _drain_external_runtime_requests(self) -> list[object]:
        if self._runtime_io is None:
            return []
        requests: list[object] = []
        requests.extend(self.supervisor.bridge.drain_task_requests())
        requests.extend(self.supervisor.bridge.drain_runtime_controls())
        return requests

    def _handle_task_request(self, request: TaskRequest) -> None:
        instruction = str(request.command_text).strip()
        if instruction == "":
            self._publish_notice(
                level="warning",
                notice="ignored empty task request",
                details={"taskId": str(request.task_id), "source": "dashboard"},
            )
            return
        try:
            self._submit_task_request(instruction, source="dashboard", task_id=str(request.task_id))
        except Exception as exc:  # noqa: BLE001
            self._publish_notice(
                level="error",
                notice="task request failed",
                details={
                    "taskId": str(request.task_id),
                    "instruction": instruction,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    def _handle_runtime_control(self, request: RuntimeControlRequest) -> None:
        action = str(request.action).strip().lower()
        if action not in {"set_idle", "cancel_interactive_task"}:
            self._publish_notice(
                level="warning",
                notice="unsupported runtime control request",
                details={"action": action},
            )
            return
        if not self._set_idle(source="dashboard"):
            self._publish_notice(
                level="warning",
                notice="set idle ignored",
                details={"reason": "no_active_task"},
            )

    def _publish_detector_capability(self) -> None:
        self._publisher_service().publish_detector_capability()

    def _publish_notice(self, *, level: str, notice: str, details: dict[str, object] | None = None) -> None:
        self._publisher_service().publish_notice(level=level, notice=notice, details=details)

    def _publish_runtime_snapshot(self, frame_idx: int) -> None:
        self._publisher_service().publish_runtime_snapshot(frame_idx=frame_idx)

    def _log_step(
        self,
        frame_idx: int,
        update: TrajectoryUpdate,
        command: ActionCommand | None,
        evaluation: CommandEvaluation,
    ) -> None:
        error_note = f" last_error={update.stats.last_error}" if update.stats.last_error != "" else ""
        command_type = command.action_type if command is not None else "none"
        distance_note = ""
        if evaluation.goal_distance_m >= 0.0:
            distance_note = f" goal_dist={evaluation.goal_distance_m:.3f}m"
        if command is not None and command.action_type == "LOOK_AT":
            distance_note = f" yaw_error={evaluation.yaw_error_rad:.3f}rad"
        route_note = ""
        snapshot = self._server.snapshot() if getattr(self, "_server", None) is not None else None
        if snapshot is not None and int(snapshot.planning.global_route.get("waypoint_count", 0) or 0) > 0:
            route_note = (
                " global_route={} wp={}/{} replan={}".format(
                    "active" if bool(snapshot.planning.global_route.get("active", False)) else "idle",
                    int(snapshot.planning.global_route.get("waypoint_index", 0) or 0),
                    int(snapshot.planning.global_route.get("waypoint_count", 0) or 0),
                    str(snapshot.planning.planner_control_reason) or "-",
                )
            )
        print(
            f"{self._log_prefix()}"
            f"[step={frame_idx}] command={command_type}{distance_note} "
            f"cmd=({float(self._command[0]):.3f},{float(self._command[1]):.3f},{float(self._command[2]):.3f}) "
            f"plan_v={update.plan_version} plan_ok={update.stats.successful_calls} "
            f"plan_fail={update.stats.failed_calls} plan_latency_ms={update.stats.latency_ms:.1f}{route_note}{error_note}"
        )
def main() -> int:
    try:
        args = build_arg_parser().parse_args()
        args = apply_demo_defaults(args)
        validate_args(args)
        args = apply_launch_mode_defaults(args)
        _trace_runtime_event(
            "args parsed "
            f"planner_mode={getattr(args, 'planner_mode', '')} "
            f"launch_mode={getattr(args, 'resolved_launch_mode', '')} "
            f"headless={bool(getattr(args, 'headless', False))}"
        )
    except ValueError as exc:
        _trace_runtime_event(f"arg validation failed: {type(exc).__name__}: {exc}")
        print(f"[G1_POINTGOAL] {exc}")
        return 2

    from isaacsim import SimulationApp
    from locomotion.runtime import run as run_g1_play

    launch_config = build_launch_config(args)
    _trace_runtime_event(f"creating SimulationApp launch_config={launch_config}")
    simulation_app = SimulationApp(launch_config=launch_config)
    _trace_runtime_event("SimulationApp created")

    try:
        _trace_runtime_event("entering locomotion runtime")
        exit_code = run_g1_play(args, simulation_app, command_source=AuraRuntimeCommandSource(args))
        _trace_runtime_event(f"locomotion runtime returned exit_code={exit_code}")
        return exit_code
    except Exception as exc:  # noqa: BLE001
        _trace_runtime_exception(exc)
        print(f"[G1_POINTGOAL] unhandled exception: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return 1
    finally:
        _trace_runtime_event("closing SimulationApp")
        simulation_app.close()
        _trace_runtime_event("SimulationApp closed")


if __name__ == "__main__":
    raise SystemExit(main())
