from __future__ import annotations

from dataclasses import replace
import threading
import time
import traceback
from typing import Any

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

from .aura_runtime_args import apply_demo_defaults, apply_launch_mode_defaults, build_arg_parser, resolve_launch_mode, validate_args
from .planning_session import PlanningSession, TrajectoryUpdate
from .subgoal_executor import CommandEvaluation, SubgoalExecutor
from .supervisor import Supervisor, SupervisorConfig


class AuraRuntimeCommandSource:
    def __init__(
        self,
        args,
        *,
        supervisor: Supervisor | None = None,
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
        self._mode = str(getattr(args, "planner_mode", "interactive")).strip().lower()
        self._executor = SubgoalExecutor(args, planning_session=planning_session or PlanningSession(args))
        self._supervisor_config = SupervisorConfig(
            memory_db_path=str(getattr(args, "memory_db_path", "state/memory/memory.sqlite")),
            detector_model_path=str(getattr(args, "detector_model_path", "")),
            detector_device=str(getattr(args, "detector_device", "")),
            memory_store=bool(getattr(args, "memory_store", True)),
            skip_detection=bool(getattr(args, "skip_detection", False)),
        )
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
        self._interactive_running = False
        self._interactive_input_thread: threading.Thread | None = None
        self._last_interactive_phase = ""
        self._last_interactive_command_id = -1
        self._last_frame_header: FrameHeader | None = None
        self._last_capture_report: dict[str, object] = {}
        self._last_viewer_overlay: dict[str, object] = {}
        self._last_sensor_meta: dict[str, object] = {}
        self._last_runtime_snapshot_frame = -1

    @property
    def supervisor(self) -> Supervisor:
        if self._supervisor is None:
            self._supervisor = Supervisor(config=self._supervisor_config)
        return self._supervisor

    @property
    def planning_session(self) -> PlanningSession:
        return self._executor.planning_session

    def initialize(self, simulation_app, stage, controller) -> None:
        self._controller = controller
        for _ in range(max(int(self.args.startup_updates), 0)):
            simulation_app.update()
        self._executor.initialize(simulation_app, stage)
        self._ensure_runtime_bridge()
        self._bootstrap_mode()
        self._publish_detector_capability()
        self._publish_notice(
            level="info",
            notice="aura runtime ready",
            details={
                "plannerMode": self._mode,
                "launchMode": self._launch_mode,
                "viewerPublish": bool(self._viewer_publish),
                "nativeViewer": self._native_viewer,
            },
        )
        if self._mode == "interactive":
            self._interactive_running = True
            self._print_interactive_help()
            self._interactive_input_thread = threading.Thread(
                target=self._interactive_input_loop,
                name="aura-runtime-stdin",
                daemon=True,
            )
            self._interactive_input_thread.start()

    def update(self, frame_idx: int) -> None:
        if self._controller is None:
            raise RuntimeError("AuraRuntimeCommandSource.initialize() must be called before update().")

        base_state = self._controller.get_base_state()
        robot_pose = tuple(float(v) for v in np.asarray(base_state.position_w, dtype=np.float32).reshape(-1)[:3])
        robot_quat = np.asarray(base_state.quat_wxyz, dtype=np.float32)
        robot_yaw = float(quat_wxyz_to_yaw(robot_quat))
        self._last_robot_pose_xyz = robot_pose
        self._drain_external_runtime_requests()

        observation = self.planning_session.capture_observation(frame_idx)
        if observation is not None:
            planner_overlay_state: dict[str, object] = {}
            planner_overlay_getter = getattr(self.planning_session, "viewer_overlay_state", None)
            if callable(planner_overlay_getter):
                raw_overlay_state = planner_overlay_getter()
                if isinstance(raw_overlay_state, dict):
                    planner_overlay_state = dict(raw_overlay_state)
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
                        "planner_overlay": planner_overlay_state,
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
            enriched = self.supervisor.process_frame(batch, publish=self._viewer_publish)
            self._last_frame_header = enriched.frame_header
            self._last_capture_report = dict(enriched.capture_report)
            self._last_sensor_meta = dict(observation.sensor_meta)
            overlay = enriched.frame_header.metadata.get("viewer_overlay", {})
            self._last_viewer_overlay = dict(overlay) if isinstance(overlay, dict) else {}
            active_memory_instruction = self.planning_session.active_memory_instruction()
            if active_memory_instruction != "":
                memory_context = self.supervisor.memory_service.build_memory_context(
                    instruction=active_memory_instruction,
                    current_pose=tuple(float(v) for v in robot_pose[:3]),
                )
                observation = replace(observation, memory_context=memory_context)

        command = self._resolve_action_command(robot_pose=robot_pose)
        execution = self._executor.step(
            frame_idx=frame_idx,
            observation=observation,
            action_command=command,
            robot_pos_world=np.asarray(base_state.position_w, dtype=np.float32),
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat,
        )
        update = execution.trajectory_update
        evaluation = execution.evaluation
        self._command = execution.command_vector
        self._pending_status = execution.status
        self._sync_planner_scratchpad(update)
        if self._pending_status is not None and self._runtime_io is not None:
            self.supervisor.bridge.publish_status(self._pending_status)
        self._publish_runtime_snapshot(frame_idx, update=update, evaluation=evaluation)

        if evaluation.reached_goal and self._mode == "pointgoal":
            self._arm_exit(0, f"goal reached at step={frame_idx} dist={evaluation.goal_distance_m:.3f}m")
        elif self._pending_status is not None and self._pending_status.state == "failed" and self._mode == "pointgoal":
            self._arm_exit(1, self._pending_status.reason or "pointgoal planning failed")

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
        if self._mode == "interactive" and update.interactive_phase == "roaming":
            log_interval = max(int(self.args.interactive_idle_log_interval), 1)
        if frame_idx % log_interval == 0:
            self._log_step(frame_idx, update, command, evaluation)

    def command(self) -> np.ndarray:
        return self._command.copy()

    def shutdown(self) -> None:
        self._interactive_running = False
        self._publish_notice(level="info", notice="aura runtime shutdown", details={"reason": self.shutdown_reason})
        self._executor.shutdown()
        if self._runtime_io is not None:
            self._runtime_io.close(unlink_shm=True)
            self._runtime_io = None

    def _bootstrap_mode(self) -> None:
        if self._mode == "pointgoal":
            goal_x = float(self.args.goal_x if self.args.goal_x is not None else 0.0)
            goal_y = float(self.args.goal_y if self.args.goal_y is not None else 0.0)
            self._manual_command = ActionCommand(
                action_type="NAV_TO_POSE",
                task_id="pointgoal",
                target_pose_xyz=(goal_x, goal_y, 0.0),
                stop_radius_m=float(getattr(self.args, "goal_tolerance_m", 0.4)),
                metadata={"source": "aura_runtime_pointgoal"},
            )
            return
        if self._mode == "dual":
            instruction = str(getattr(self.args, "instruction", "")).strip()
            if instruction != "":
                self.planning_session.ensure_navdp_service_ready(context="dual startup")
                self.planning_session.ensure_dual_service_ready(context="dual startup")
                self.planning_session.start_dual_task(instruction)
                self.supervisor.memory_service.set_planner_task(
                    instruction=instruction,
                    planner_mode="dual",
                    task_state="active",
                    task_id="dual",
                )
                self._manual_command = self._build_planner_managed_command(task_id="dual", source="aura_runtime_dual")
            return
        if self._mode == "interactive":
            self.planning_session.ensure_navdp_service_ready(context="interactive startup")
            self._manual_command = self._build_planner_managed_command(task_id="interactive", source="aura_runtime_interactive")

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

    def _print_interactive_help(self) -> None:
        print("[G1_INTERACTIVE][ROAM] terminal natural-language control")
        print("[G1_INTERACTIVE][ROAM]   text      : submit a System2 navigation request")
        print("[G1_INTERACTIVE][ROAM]   /pointgoal x y : submit a System1-only world-frame point goal")
        print("[G1_INTERACTIVE][ROAM]   /help     : show this help")
        print("[G1_INTERACTIVE][ROAM]   /cancel   : cancel the active task and resume roaming")
        print("[G1_INTERACTIVE][ROAM]   /quit     : exit the runtime")

    def _interactive_input_loop(self) -> None:
        prompt = str(getattr(self.args, "interactive_prompt", "nl>")).strip() or "nl>"
        while self._interactive_running and not self.quit_requested:
            try:
                raw = input(f"{prompt} ")
            except EOFError:
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[G1_INTERACTIVE][ROAM] input loop stopped: {type(exc).__name__}: {exc}")
                return

            text = raw.strip()
            if text == "":
                continue

            lowered = text.lower()
            if lowered == "/help":
                self._print_interactive_help()
                continue
            if lowered == "/cancel":
                cancelled = self._cancel_interactive_task(source="stdin")
                if cancelled:
                    print("[G1_INTERACTIVE][TASK] cancel requested")
                else:
                    print("[G1_INTERACTIVE][ROAM] no active task to cancel")
                continue
            if lowered == "/quit":
                self._arm_exit(0, "interactive quit requested")
                return
            if lowered.startswith("/") and self._parse_interactive_pointgoal_command(text) is None:
                print(f"[G1_INTERACTIVE][ROAM] unknown command: {text}")
                continue

            try:
                command_id, queued_label = self._submit_interactive_request(text, source="stdin")
            except Exception as exc:  # noqa: BLE001
                print(f"[G1_INTERACTIVE][TASK] rejected instruction={text!r} error={type(exc).__name__}: {exc}")
                continue
            print(f"[G1_INTERACTIVE][TASK] command_id={command_id} queued instruction={queued_label!r}")

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
            except RuntimeError:
                bus = InprocBus()
            self._runtime_io = RuntimeIo(
                bus=bus,
                shm_ring=None,
            )
        self._supervisor = Supervisor(
            bus=self._runtime_io.bus,
            shm_ring=self._runtime_io.shm_ring,
            config=self._supervisor_config,
        )
        print(
            "[AURA_RUNTIME] runtime bridge ready "
            f"control={getattr(self.args, 'viewer_control_endpoint', '')} "
            f"telemetry={getattr(self.args, 'viewer_telemetry_endpoint', '')} "
            f"shm={getattr(self.args, 'viewer_shm_name', '') if self._viewer_publish else 'disabled'} "
            f"viewer_publish={self._viewer_publish}"
        )

    def _log_prefix(self) -> str:
        if self._mode == "pointgoal":
            return "[G1_POINTGOAL]"
        if self._mode == "dual":
            return "[G1_DUAL]"
        if self._mode == "interactive":
            return "[G1_INTERACTIVE]"
        return "[G1_DIRECT]"

    def _build_planner_managed_command(self, *, task_id: str, source: str) -> ActionCommand:
        return ActionCommand(
            action_type="LOCAL_SEARCH",
            task_id=task_id,
            metadata={
                "source": source,
                "planner_managed": True,
                "planner_mode": self._mode,
            },
        )

    @staticmethod
    def _parse_interactive_pointgoal_command(text: str) -> tuple[float, float] | None:
        tokens = str(text).strip().split()
        if len(tokens) == 0:
            return None
        command = tokens[0].strip().lower()
        if command not in {"/pointgoal", "/pg"}:
            return None
        if len(tokens) != 3:
            raise ValueError("point goal command format: /pointgoal <x> <y>")
        try:
            goal_x = float(tokens[1])
            goal_y = float(tokens[2])
        except ValueError as exc:
            raise ValueError("point goal command requires numeric x and y values") from exc
        return goal_x, goal_y

    def _submit_interactive_request(self, text: str, *, source: str, task_id: str = "") -> tuple[int, str]:
        if self._mode != "interactive":
            raise RuntimeError("interactive instruction requires planner-mode=interactive")
        raw_text = str(text).strip()
        if raw_text == "":
            raise ValueError("interactive instruction must be non-empty")
        pointgoal = self._parse_interactive_pointgoal_command(raw_text)
        if pointgoal is not None:
            goal_x, goal_y = pointgoal
            label = f"/pointgoal {goal_x:.3f} {goal_y:.3f}"
            self.planning_session.ensure_navdp_service_ready(context=f"interactive pointgoal ({source})")
            command_id = int(
                self.planning_session.submit_interactive_point_goal(
                    (goal_x, goal_y),
                    label=label,
                )
            )
            self.supervisor.memory_service.set_planner_task(
                instruction=label,
                planner_mode="interactive",
                task_state="pending",
                task_id=str(task_id or "interactive"),
                command_id=command_id,
            )
            self._publish_notice(
                level="info",
                notice="interactive point goal queued",
                details={
                    "source": source,
                    "taskId": str(task_id or "interactive"),
                    "commandId": command_id,
                    "goal": {"x": goal_x, "y": goal_y},
                },
            )
            return command_id, label
        instruction = raw_text
        self.planning_session.ensure_navdp_service_ready(context=f"interactive task ({source})")
        self.planning_session.ensure_dual_service_ready(context=f"interactive task ({source})")
        command_id = int(self.planning_session.submit_interactive_instruction(instruction))
        self.supervisor.memory_service.set_planner_task(
            instruction=instruction,
            planner_mode="interactive",
            task_state="pending",
            task_id=str(task_id or "interactive"),
            command_id=command_id,
        )
        self._publish_notice(
            level="info",
            notice="interactive task queued",
            details={
                "source": source,
                "taskId": str(task_id or "interactive"),
                "commandId": command_id,
                "instruction": instruction,
            },
        )
        return command_id, instruction

    def _submit_interactive_instruction(self, text: str, *, source: str, task_id: str = "") -> int:
        command_id, _queued_label = self._submit_interactive_request(text, source=source, task_id=task_id)
        return command_id

    def _cancel_interactive_task(self, *, source: str) -> bool:
        cancelled = bool(self.planning_session.cancel_interactive_task())
        if cancelled:
            self.supervisor.memory_service.clear_planner_task(
                task_state="cancelled",
                reason=f"interactive task cancelled via {source}",
            )
            self._publish_notice(
                level="info",
                notice="interactive task cancelled",
                details={"source": source},
            )
        return cancelled

    def _drain_external_runtime_requests(self) -> None:
        if self._runtime_io is None:
            return
        for request in self.supervisor.bridge.drain_task_requests():
            self._handle_task_request(request)
        for request in self.supervisor.bridge.drain_runtime_controls():
            self._handle_runtime_control(request)

    def _handle_task_request(self, request: TaskRequest) -> None:
        instruction = str(request.command_text).strip()
        if instruction == "":
            self._publish_notice(
                level="warning",
                notice="ignored empty task request",
                details={"taskId": str(request.task_id), "source": "dashboard"},
            )
            return
        if self._mode != "interactive":
            self._publish_notice(
                level="warning",
                notice="task request rejected",
                details={
                    "reason": "planner_mode_not_interactive",
                    "plannerMode": self._mode,
                    "taskId": str(request.task_id),
                    "instruction": instruction,
                },
            )
            return
        try:
            self._submit_interactive_request(instruction, source="dashboard", task_id=str(request.task_id))
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
        if action != "cancel_interactive_task":
            self._publish_notice(
                level="warning",
                notice="unsupported runtime control request",
                details={"action": action},
            )
            return
        if not self._cancel_interactive_task(source="dashboard"):
            self._publish_notice(
                level="warning",
                notice="interactive cancel ignored",
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
        if self._runtime_io is None:
            return
        self.supervisor.bridge.publish_notice(
            RuntimeNotice(component="aura_runtime", level=level, notice=notice, details=dict(details or {}))
        )

    def _publish_runtime_snapshot(
        self,
        frame_idx: int,
        *,
        update: TrajectoryUpdate,
        evaluation: CommandEvaluation,
    ) -> None:
        if self._runtime_io is None:
            return
        interval = max(int(getattr(self.args, "log_interval", 30)), 1)
        if self._last_runtime_snapshot_frame >= 0 and (frame_idx - self._last_runtime_snapshot_frame) < interval:
            return
        self._last_runtime_snapshot_frame = int(frame_idx)
        self.supervisor.bridge.publish_health(
            HealthPing(
                component="aura_runtime",
                details={"snapshot": self._build_runtime_snapshot(update=update, evaluation=evaluation)},
            )
        )

    def _build_runtime_snapshot(
        self,
        *,
        update: TrajectoryUpdate,
        evaluation: CommandEvaluation,
    ) -> dict[str, object]:
        detector = self.supervisor.perception_pipeline.detector
        detector_report = detector.runtime_report
        scratchpad = self.supervisor.memory_service.scratchpad
        frame_header = self._last_frame_header
        overlay = self._last_viewer_overlay if isinstance(self._last_viewer_overlay, dict) else {}
        detections = overlay.get("detections", [])
        transport = {
            "viewerPublish": bool(self._viewer_publish),
            "nativeViewer": self._native_viewer,
            "controlEndpoint": str(getattr(self.args, "viewer_control_endpoint", "")),
            "telemetryEndpoint": str(getattr(self.args, "viewer_telemetry_endpoint", "")),
            "shmName": str(getattr(self.args, "viewer_shm_name", "")),
            "frameAvailable": frame_header is not None,
        }
        modes = {
            "plannerMode": self._mode,
            "launchMode": self._launch_mode,
            "viewerPublish": bool(self._viewer_publish),
            "nativeViewer": self._native_viewer,
            "scenePreset": str(getattr(self.args, "scene_preset", "")),
            "showDepth": bool(getattr(self.args, "show_depth", False)),
            "memoryStore": bool(getattr(self.args, "memory_store", True)),
            "detectionEnabled": not bool(getattr(self.args, "skip_detection", False)),
        }
        planner = {
            "planVersion": int(update.plan_version),
            "goalVersion": int(update.goal_version),
            "trajVersion": int(update.traj_version),
            "staleSec": float(update.stale_sec),
            "plannerControlMode": str(update.planner_control_mode),
            "plannerYawDeltaRad": float(update.planner_yaw_delta_rad) if update.planner_yaw_delta_rad is not None else None,
            "goalDistanceM": float(evaluation.goal_distance_m),
            "yawErrorRad": float(evaluation.yaw_error_rad),
            "interactivePhase": str(update.interactive_phase or ""),
            "interactiveCommandId": int(update.interactive_command_id),
            "interactiveInstruction": str(update.interactive_instruction),
            "actionStatus": None
            if self._pending_status is None
            else {
                "state": str(self._pending_status.state),
                "success": bool(self._pending_status.success),
                "reason": str(self._pending_status.reason),
                "distanceRemainingM": self._pending_status.distance_remaining_m,
            },
            "activeCommandType": "" if self._active_command is None else str(self._active_command.action_type),
        }
        sensor = {
            "rgbAvailable": frame_header is not None,
            "depthAvailable": frame_header is not None and bool(overlay.get("has_depth", False) or "depth_ref" in frame_header.metadata or "depth_inline" in frame_header.metadata),
            "poseAvailable": frame_header is not None,
            "frameId": None if frame_header is None else int(frame_header.frame_id),
            "source": "" if frame_header is None else str(frame_header.source),
            "cameraPoseXyz": [] if frame_header is None else [float(v) for v in frame_header.camera_pose_xyz[:3]],
            "robotPoseXyz": [] if frame_header is None else [float(v) for v in frame_header.robot_pose_xyz[:3]],
            "robotYawRad": None if frame_header is None else float(frame_header.robot_yaw_rad),
            "sensorMeta": dict(self._last_sensor_meta),
            "captureReport": dict(self._last_capture_report),
        }
        perception = {
            "detectorBackend": str(detector.info.backend_name),
            "detectorSelectedReason": str(detector.info.selected_reason),
            "detectorReady": bool(detector_report.ready_for_inference) if detector_report is not None else False,
            "detectorRuntimeReport": None if detector_report is None else detector_report.as_dict(),
            "detectionCount": len(detections) if isinstance(detections, list) else 0,
            "trackedDetectionCount": len(detections) if isinstance(detections, list) else 0,
            "trajectoryPointCount": len(overlay.get("trajectory_pixels", [])) if isinstance(overlay.get("trajectory_pixels", []), list) else 0,
        }
        memory = {
            "objectCount": len(self.supervisor.memory_service.spatial_store.objects),
            "placeCount": len(self.supervisor.memory_service.spatial_store.places),
            "semanticRuleCount": len(self.supervisor.memory_service.semantic_store.list()),
            "keyframeCount": len(self.supervisor.memory_service.keyframes),
            "scratchpad": {
                "instruction": str(scratchpad.instruction),
                "plannerMode": str(scratchpad.planner_mode),
                "taskState": str(scratchpad.task_state),
                "taskId": str(scratchpad.task_id),
                "commandId": int(scratchpad.command_id),
                "goalSummary": str(scratchpad.goal_summary),
                "recentHint": str(scratchpad.recent_hint),
                "nextPriority": str(scratchpad.next_priority),
            },
            "memoryAwareTaskActive": bool(
                self._mode == "interactive" and str(update.interactive_phase or "") == "task_active" and str(update.interactive_instruction).strip() != ""
            ),
        }
        return {
            "modes": modes,
            "planner": planner,
            "sensor": sensor,
            "perception": perception,
            "memory": memory,
            "transport": transport,
        }

    def _sync_planner_scratchpad(self, update: TrajectoryUpdate) -> None:
        if self._mode == "interactive":
            current_phase = str(update.interactive_phase or "")
            current_command_id = int(update.interactive_command_id)
            if current_phase in {"task_active", "pointgoal_active"} and update.interactive_instruction.strip() != "":
                if self._last_interactive_phase != current_phase or self._last_interactive_command_id != current_command_id:
                    self.supervisor.memory_service.set_planner_task(
                        instruction=update.interactive_instruction,
                        planner_mode="interactive",
                        task_state="active",
                        task_id="interactive",
                        command_id=current_command_id,
                    )
            elif self._last_interactive_phase in {"task_active", "pointgoal_active"} and current_phase == "roaming":
                clear_state = "completed" if bool(update.stop) else "idle"
                if self._last_interactive_phase == "pointgoal_active":
                    clear_reason = "interactive point goal complete" if bool(update.stop) else "interactive point goal cleared"
                else:
                    clear_reason = "interactive task complete" if bool(update.stop) else "interactive task cleared"
                if update.stats.last_error != "":
                    clear_state = "failed"
                    clear_reason = str(update.stats.last_error)
                self.supervisor.memory_service.clear_planner_task(
                    task_state=clear_state,
                    reason=clear_reason,
                )
            self._last_interactive_phase = current_phase
            self._last_interactive_command_id = current_command_id
            return

        if self._mode == "dual" and bool(update.stop) and update.planner_control_mode == "stop":
            self.supervisor.memory_service.clear_planner_task(
                task_state="completed",
                reason="dual task complete",
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
        print(
            f"{self._log_prefix()}"
            f"[step={frame_idx}] command={command_type}{distance_note} "
            f"cmd=({float(self._command[0]):.3f},{float(self._command[1]):.3f},{float(self._command[2]):.3f}) "
            f"plan_v={update.plan_version} plan_ok={update.stats.successful_calls} "
            f"plan_fail={update.stats.failed_calls} plan_latency_ms={update.stats.latency_ms:.1f}{error_note}"
        )


def build_launch_config(args) -> dict[str, bool]:
    launch_config = {"headless": bool(args.headless)}
    viewer_publish = bool(getattr(args, "viewer_publish", False))
    if bool(args.headless) and not viewer_publish:
        launch_config["disable_viewport_updates"] = True
    return launch_config


def main() -> int:
    try:
        args = build_arg_parser().parse_args()
        args = apply_demo_defaults(args)
        validate_args(args)
        args = apply_launch_mode_defaults(args)
    except ValueError as exc:
        print(f"[G1_POINTGOAL] {exc}")
        return 2

    from isaacsim import SimulationApp
    from locomotion.runtime import run as run_g1_play

    launch_config = build_launch_config(args)
    simulation_app = SimulationApp(launch_config=launch_config)

    try:
        return run_g1_play(args, simulation_app, command_source=AuraRuntimeCommandSource(args))
    except Exception as exc:  # noqa: BLE001
        print(f"[G1_POINTGOAL] unhandled exception: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
