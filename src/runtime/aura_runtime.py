from __future__ import annotations

import os
import time
import traceback
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from apps.runtime_common import RuntimeIo, build_runtime_io
from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from common.geometry import quat_wxyz_to_yaw
from ipc.inproc_bus import InprocBus
from ipc.messages import (
    ActionCommand,
    ActionStatus,
    CapabilityReport,
    FrameHeader,
    HealthPing,
    RuntimeControlRequest,
    RuntimeNotice,
    TaskRequest,
)
from ipc.zmq_bus import ZmqBus
from schemas.events import FrameEvent, WorkerMetadata
from server.snapshot_adapter import SnapshotAdapter

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
        self._controller = controller
        for _ in range(max(int(self.args.startup_updates), 0)):
            simulation_app.update()
        self._ensure_runtime_bridge()
        self._ensure_control_server()
        assert self._server is not None
        self._server.initialize(simulation_app, stage)
        self._bootstrap_mode()
        self._seed_planner_overlay()
        self._publish_detector_capability()
        self._publish_notice(
            level="info",
            notice="aura runtime ready",
            details={
                "launchMode": self._launch_mode,
                "viewerPublish": bool(self._viewer_publish),
                "nativeViewer": self._native_viewer,
            },
        )

    def update(self, frame_idx: int) -> None:
        if self._controller is None:
            raise RuntimeError("AuraRuntimeCommandSource.initialize() must be called before update().")
        self._seed_planner_overlay()

        base_state = self._controller.get_base_state()
        robot_pose = tuple(float(v) for v in np.asarray(base_state.position_w, dtype=np.float32).reshape(-1)[:3])
        robot_quat = np.asarray(base_state.quat_wxyz, dtype=np.float32)
        robot_yaw = float(quat_wxyz_to_yaw(robot_quat))
        self._last_robot_pose_xyz = robot_pose
        runtime_events = self._drain_external_runtime_requests()

        observation = self.planning_session.capture_observation(frame_idx)
        batch = None
        if observation is not None:
            batch = IsaacObservationBatch(
                frame_header=FrameHeader(
                    frame_id=int(observation.frame_id),
                    timestamp_ns=time.time_ns(),
                    source="aura_runtime",
                    width=int(observation.rgb.shape[1]),
                    height=int(observation.rgb.shape[0]),
                    camera_pose_xyz=tuple(float(v) for v in observation.cam_pos[:3]),
                    camera_quat_wxyz=tuple(float(v) for v in observation.cam_quat[:4]),
                    robot_pose_xyz=robot_pose,
                    robot_yaw_rad=float(robot_yaw),
                    sim_time_s=float(time.time()),
                    metadata={
                        **dict(observation.sensor_meta),
                        "planner_overlay": dict(self._last_viewer_overlay),
                        "active_command_overlay": self._command_overlay_metadata(),
                    },
                ),
                robot_pose_xyz=robot_pose,
                robot_yaw_rad=float(robot_yaw),
                sim_time_s=float(time.time()),
                rgb_image=observation.rgb,
                depth_image_m=observation.depth,
                camera_intrinsic=observation.intrinsic,
                capture_report=dict(observation.sensor_meta),
            )
        frame_event = FrameEvent(
            metadata=WorkerMetadata(
                task_id=self.supervisor.memory_service.scratchpad.task_id,
                frame_id=int(frame_idx),
                timestamp_ns=time.time_ns(),
                source="aura_runtime",
                timeout_ms=int(max(float(getattr(self.args, "timeout_sec", 5.0)) * 1000.0, 0.0)),
            ),
            frame_id=int(frame_idx),
            timestamp_ns=time.time_ns(),
            source="aura_runtime",
            robot_pose_xyz=robot_pose,
            robot_yaw_rad=float(robot_yaw),
            sim_time_s=float(time.time()),
            observation=observation,
            batch=batch,
            sensor_meta={} if observation is None else dict(observation.sensor_meta),
            planner_overlay={} if observation is None else dict(batch.frame_header.metadata.get("planner_overlay", {})),
            publish_observation=bool(self._viewer_publish),
        )
        self._ensure_control_server()
        assert self._server is not None
        incoming_status = self._pending_status
        tick_result = self._server.tick(
            frame_event=frame_event,
            task_events=runtime_events,
            runtime_status=incoming_status,
            robot_pos_world=np.asarray(base_state.position_w, dtype=np.float32),
            robot_lin_vel_world=np.asarray(base_state.lin_vel_w, dtype=np.float32),
            robot_ang_vel_world=np.asarray(base_state.ang_vel_w, dtype=np.float32),
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat,
        )
        for notice in tick_result.notices:
            self.supervisor.bridge.publish_notice(notice)
        update = tick_result.trajectory_update
        evaluation = tick_result.evaluation
        self._command = tick_result.command_vector.copy()
        self._pending_status = tick_result.status
        self._active_command = tick_result.action_command
        self._last_viewer_overlay = dict(tick_result.viewer_overlay)
        if self._pending_status is not None and self._runtime_io is not None:
            self.supervisor.bridge.publish_status(self._pending_status)
        self._publish_runtime_snapshot(frame_idx)

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
            self.supervisor.bridge.publish_notice(notice)

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
        if self._runtime_io is None:
            return
        detector_report = self.supervisor.perception_pipeline.detector.runtime_report
        if detector_report is None:
            return
        self.supervisor.bridge.publish_capability(
            CapabilityReport(
                component="detector",
                status="ready" if detector_report.ready_for_inference else "fallback",
                backend_name=self.supervisor.perception_pipeline.detector.info.backend_name,
                details=detector_report.as_dict(),
                warnings=list(detector_report.warnings),
                errors=list(detector_report.errors),
            )
        )

    def _publish_notice(self, *, level: str, notice: str, details: dict[str, object] | None = None) -> None:
        if self._runtime_io is None or self._supervisor is None:
            return
        self._supervisor.bridge.publish_notice(
            RuntimeNotice(component="aura_runtime", level=level, notice=notice, details=dict(details or {}))
        )

    def _publish_runtime_snapshot(self, frame_idx: int) -> None:
        if self._runtime_io is None:
            return
        interval = max(int(getattr(self.args, "log_interval", 30)), 1)
        if self._last_runtime_snapshot_frame >= 0 and (frame_idx - self._last_runtime_snapshot_frame) < interval:
            return
        self._last_runtime_snapshot_frame = int(frame_idx)
        snapshot = None if self._server is None else self._server.snapshot()
        self.supervisor.bridge.publish_health(
            HealthPing(
                component="aura_runtime",
                details={
                    "worldState": {} if snapshot is None else snapshot.to_dict(),
                    "snapshot": SnapshotAdapter.to_legacy_runtime_payload(snapshot),
                },
            )
        )

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
