from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter, IsaacBridgeAdapterConfig, IsaacObservationBatch
from adapters.sensors.isaac_live_source import IsaacLiveFrameSource, IsaacLiveSourceConfig
from common.geometry import quat_wxyz_to_yaw
from ipc.base import MessageBus
from ipc.messages import ActionCommand, CapabilityReport, FrameHeader, HealthPing, RuntimeNotice, TaskRequest
from ipc.shm_ring import SharedMemoryRing

from .planning_session import ExecutionObservation, PlanningSession
from .subgoal_executor import SubgoalExecutionResult, SubgoalExecutor


class FrameBridgeCommandSource:
    def __init__(
        self,
        args,
        *,
        bus: MessageBus,
        shm_ring: SharedMemoryRing | None = None,
        planning_session: PlanningSession | None = None,
    ) -> None:
        self.args = args
        self.quit_requested = False
        self.exit_code = 0
        self.shutdown_reason = ""
        self.requires_render = bool(getattr(args, "headless", False))
        self._controller = None
        self._bridge = IsaacBridgeAdapter(bus, IsaacBridgeAdapterConfig(), shm_ring=shm_ring)
        self._executor = SubgoalExecutor(args, planning_session=planning_session)
        self._frame_source: IsaacLiveFrameSource | None = None
        self._command = np.zeros(3, dtype=np.float32)
        self._active_command: ActionCommand | None = None
        self._last_robot_pose_xyz = (0.0, 0.0, 0.0)
        self._last_robot_yaw = 0.0
        self._health_period = max(int(getattr(args, "log_interval", 30)), 1)
        self._sensor_report_path = str(getattr(args, "sensor_report_path", "")).strip()

    @property
    def planning_session(self) -> PlanningSession:
        return self._executor.planning_session

    def initialize(self, simulation_app, stage, controller) -> None:
        self._controller = controller
        for _ in range(max(int(getattr(self.args, "startup_updates", 0)), 0)):
            simulation_app.update()
        self._executor.initialize(simulation_app, stage)
        self._frame_source = IsaacLiveFrameSource(
            sensor_adapter=self.planning_session.sensor,
            robot_pose_provider=lambda: self._last_robot_pose_xyz,
            robot_yaw_provider=lambda: self._last_robot_yaw,
            config=IsaacLiveSourceConfig(
                source_name="frame_bridge_live",
                strict_live=bool(getattr(self.args, "strict_live", False)),
                image_width=int(getattr(self.args, "image_width", 640)),
                image_height=int(getattr(self.args, "image_height", 640)),
                depth_max_m=float(getattr(self.args, "depth_max_m", 5.0)),
            ),
        )
        report = self._frame_source.start()
        self._publish_startup_health()
        self._publish_sensor_diagnostics(report.notice)
        if report.status != "ready":
            raise RuntimeError(report.notice or "live frame source unavailable")
        self._bridge.publish_notice(
            RuntimeNotice(
                component="frame_bridge",
                level="warning" if report.fallback_used else "info",
                notice=report.notice or f"live frame source status={report.status}",
                details={
                    "frame_source": report.source_name,
                    "live_available": bool(report.live_available),
                    "fallback_used": bool(report.fallback_used),
                    **dict(report.details),
                },
            )
        )
        command_text = str(getattr(self.args, "command", "")).strip()
        if command_text != "":
            self._bridge.publish_task_request(TaskRequest(command_text=command_text))

    def update(self, frame_idx: int) -> None:
        if self._controller is None:
            raise RuntimeError("FrameBridgeCommandSource.initialize() must be called before update().")

        base_state = self._controller.get_base_state()
        robot_pos_world = np.asarray(base_state.position_w, dtype=np.float32)
        robot_quat = np.asarray(base_state.quat_wxyz, dtype=np.float32)
        robot_pose = tuple(float(v) for v in robot_pos_world.reshape(-1)[:3])
        robot_yaw = float(quat_wxyz_to_yaw(robot_quat))
        self._last_robot_pose_xyz = robot_pose
        self._last_robot_yaw = robot_yaw

        observation = self._publish_live_observation(frame_idx=frame_idx, robot_pose=robot_pose, robot_yaw=robot_yaw)
        records = self._bridge.drain_commands()
        if records:
            self._active_command = records[-1]

        execution = self._executor.step(
            frame_idx=frame_idx,
            observation=observation,
            action_command=self._active_command,
            robot_pos_world=robot_pos_world,
            robot_lin_vel_world=np.asarray(base_state.lin_vel_w, dtype=np.float32),
            robot_ang_vel_world=np.asarray(base_state.ang_vel_w, dtype=np.float32),
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat,
        )
        self._command = execution.command_vector
        self._publish_status(execution)

        if execution.status is not None and execution.status.state in {"succeeded", "failed"}:
            self._active_command = None

        if frame_idx % self._health_period == 0:
            self._publish_health(frame_idx=frame_idx)
            self._log_step(frame_idx=frame_idx, execution=execution)

    def command(self) -> np.ndarray:
        return self._command.copy()

    def shutdown(self) -> None:
        if self._frame_source is not None:
            self._frame_source.close()
        self._executor.shutdown()

    def _publish_live_observation(
        self,
        *,
        frame_idx: int,
        robot_pose: tuple[float, float, float],
        robot_yaw: float,
    ) -> ExecutionObservation | None:
        if self._frame_source is not None:
            sample = self._frame_source.read()
            if sample is not None:
                sample.frame_id = int(frame_idx)
                batch = self._frame_sample_to_batch(sample)
                self._bridge.publish_observation_batch(batch)
                return self.planning_session.build_local_observation(
                    frame_id=int(frame_idx),
                    rgb=batch.rgb_image if batch.rgb_image is not None else np.zeros((0, 0, 3), dtype=np.uint8),
                    depth=batch.depth_image_m if batch.depth_image_m is not None else np.zeros((0, 0), dtype=np.float32),
                    camera_pose_xyz=batch.frame_header.camera_pose_xyz,
                    camera_quat_wxyz=batch.frame_header.camera_quat_wxyz,
                    intrinsic=batch.camera_intrinsic,
                    sensor_meta={
                        **dict(batch.frame_header.metadata),
                        "capture_report": dict(batch.capture_report),
                    },
                )
        observation = self.planning_session.capture_observation(frame_idx)
        if observation is None:
            return None
        self._bridge.publish_observation_batch(
            IsaacObservationBatch(
                frame_header=FrameHeader(
                    frame_id=int(observation.frame_id),
                    timestamp_ns=time.time_ns(),
                    source="frame_bridge_sensor",
                    width=int(observation.rgb.shape[1]),
                    height=int(observation.rgb.shape[0]),
                    camera_pose_xyz=tuple(float(v) for v in observation.cam_pos[:3]),
                    camera_quat_wxyz=tuple(float(v) for v in observation.cam_quat[:4]),
                    robot_pose_xyz=robot_pose,
                    robot_yaw_rad=float(robot_yaw),
                    sim_time_s=self._sim_time_seconds(frame_idx),
                    metadata=dict(observation.sensor_meta),
                ),
                robot_pose_xyz=robot_pose,
                robot_yaw_rad=float(robot_yaw),
                sim_time_s=self._sim_time_seconds(frame_idx),
                rgb_image=observation.rgb,
                depth_image_m=observation.depth,
                camera_intrinsic=observation.intrinsic,
                capture_report=dict(observation.sensor_meta),
            )
        )
        return observation

    def _publish_startup_health(self) -> None:
        self._bridge.publish_health(
            HealthPing(
                component="frame_bridge",
                details={
                    "bus_topics": {
                        "observation": self._bridge.config.observation_topic,
                        "command": self._bridge.config.command_topic,
                        "status": self._bridge.config.status_topic,
                    }
                },
            )
        )

    def _publish_sensor_diagnostics(self, frame_source_notice: str) -> None:
        sensor = self.planning_session.sensor
        init_report = dict(self.planning_session.last_sensor_init_report)
        capture_report = {} if sensor is None else dict(sensor.last_capture_meta)
        details = {
            **init_report,
            "capture_report": capture_report,
            "frame_source_notice": str(frame_source_notice),
        }
        has_full_camera = bool(sensor is not None and sensor.rgb_prim_path and sensor.depth_prim_path)
        status = "ready" if has_full_camera else "fallback"
        print(
            "[FRAME_BRIDGE] "
            f"sensor_status={status} runtime_mount={details.get('runtime_mount', False)} "
            f"rgb={details.get('camera_prim_path', '') or capture_report.get('camera_prim_path', '')} "
            f"depth={details.get('depth_camera_prim_path', '') or capture_report.get('depth_camera_prim_path', '')}"
        )
        self._write_sensor_report(status=status, details=details)
        self._bridge.publish_capability(
            CapabilityReport(
                component="sensor",
                status=status,
                backend_name="d455",
                details=details,
                warnings=[] if status == "ready" else ["sensor is using a fallback or partial camera configuration"],
            )
        )
        self._bridge.publish_notice(
            RuntimeNotice(
                component="frame_bridge",
                level="info" if status == "ready" else "warning",
                notice=str(init_report.get("message", "sensor initialized")),
                details=details,
            )
        )

    def _write_sensor_report(self, *, status: str, details: dict[str, object]) -> None:
        if self._sensor_report_path == "":
            return
        path = Path(self._sensor_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": str(status),
                    "details": details,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _publish_status(self, execution: SubgoalExecutionResult) -> None:
        if execution.status is None:
            return
        self._bridge.publish_status(execution.status)

    def _publish_health(self, *, frame_idx: int) -> None:
        self._bridge.publish_health(
            HealthPing(
                component="frame_bridge",
                details={
                    "frame_idx": int(frame_idx),
                    "robot_pose_xyz": list(self._last_robot_pose_xyz),
                    "active_command": None if self._active_command is None else self._active_command.action_type,
                },
            )
        )

    def _log_step(self, *, frame_idx: int, execution: SubgoalExecutionResult) -> None:
        command_type = "none" if self._active_command is None else self._active_command.action_type
        evaluation = execution.evaluation
        distance_note = ""
        if evaluation.goal_distance_m >= 0.0:
            distance_note = f" goal_dist={evaluation.goal_distance_m:.3f}m"
        if self._active_command is not None and self._active_command.action_type == "LOOK_AT":
            distance_note = f" yaw_error={evaluation.yaw_error_rad:.3f}rad"
        print(
            "[FRAME_BRIDGE] "
            f"step={frame_idx} command={command_type}{distance_note} "
            f"cmd=({float(self._command[0]):.3f},{float(self._command[1]):.3f},{float(self._command[2]):.3f}) "
            f"status={None if execution.status is None else execution.status.state}"
        )

    def _frame_sample_to_batch(self, sample) -> IsaacObservationBatch:  # noqa: ANN001
        rgb = np.asarray(sample.rgb, dtype=np.uint8)
        depth = np.asarray(sample.depth, dtype=np.float32)
        metadata = dict(sample.metadata)
        capture_report = {}
        raw_capture_report = metadata.get("capture_report")
        if isinstance(raw_capture_report, dict):
            capture_report = dict(raw_capture_report)
        return IsaacObservationBatch(
            frame_header=FrameHeader(
                frame_id=int(sample.frame_id),
                timestamp_ns=time.time_ns(),
                source=str(sample.source_name),
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
                camera_pose_xyz=tuple(float(v) for v in sample.camera_pose_xyz[:3]),
                camera_quat_wxyz=tuple(float(v) for v in sample.camera_quat_wxyz[:4]),
                robot_pose_xyz=tuple(float(v) for v in sample.robot_pose_xyz[:3]),
                robot_yaw_rad=float(sample.robot_yaw_rad),
                sim_time_s=float(sample.sim_time_s),
                metadata=metadata,
            ),
            robot_pose_xyz=tuple(float(v) for v in sample.robot_pose_xyz[:3]),
            robot_yaw_rad=float(sample.robot_yaw_rad),
            sim_time_s=float(sample.sim_time_s),
            rgb_image=rgb,
            depth_image_m=depth,
            camera_intrinsic=np.asarray(sample.camera_intrinsic, dtype=np.float32),
            speaker_events=list(sample.speaker_events),
            capture_report=capture_report,
        )

    def _sim_time_seconds(self, frame_idx: int) -> float:
        return float(frame_idx) * float(getattr(self.args, "physics_dt", 1.0 / 60.0))


def run_live_frame_bridge(
    args,
    *,
    bus: MessageBus,
    shm_ring: SharedMemoryRing | None = None,
) -> int:
    from isaacsim import SimulationApp
    from locomotion.runtime import run as run_locomotion_runtime

    launch_config = {"headless": bool(getattr(args, "headless", False))}
    if bool(getattr(args, "headless", False)):
        launch_config["disable_viewport_updates"] = True
    simulation_app = SimulationApp(launch_config=launch_config)
    try:
        return run_locomotion_runtime(
            args,
            simulation_app,
            command_source=FrameBridgeCommandSource(args, bus=bus, shm_ring=shm_ring),
        )
    finally:
        simulation_app.close()
