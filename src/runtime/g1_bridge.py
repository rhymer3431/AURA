from __future__ import annotations

from dataclasses import replace
import threading
import time
import traceback

import numpy as np

from apps.runtime_common import RuntimeIo, build_runtime_io
from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from common.geometry import quat_wxyz_to_yaw
from ipc.messages import ActionCommand, ActionStatus, FrameHeader

from .g1_bridge_args import apply_demo_defaults, apply_launch_mode_defaults, build_arg_parser, resolve_launch_mode, validate_args
from .planning_session import PlanningSession, TrajectoryUpdate
from .subgoal_executor import CommandEvaluation, SubgoalExecutor
from .supervisor import Supervisor, SupervisorConfig


class NavDPCommandSource:
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
        self.requires_render = self._launch_mode in {"g1_view", "headless"}

        self._controller = None
        self._mode = str(getattr(args, "planner_mode", "interactive")).strip().lower()
        self._executor = SubgoalExecutor(args, planning_session=planning_session or PlanningSession(args))
        self._supervisor_config = SupervisorConfig(
            memory_db_path=str(getattr(args, "memory_db_path", "state/memory/memory.sqlite")),
            detector_model_path=str(getattr(args, "detector_model_path", "")),
            detector_device=str(getattr(args, "detector_device", "")),
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
        self._ensure_view_runtime()
        self._bootstrap_mode()
        if self._mode == "interactive":
            self._interactive_running = True
            self._print_interactive_help()
            self._interactive_input_thread = threading.Thread(
                target=self._interactive_input_loop,
                name="g1-interactive-stdin",
                daemon=True,
            )
            self._interactive_input_thread.start()

    def update(self, frame_idx: int) -> None:
        if self._controller is None:
            raise RuntimeError("NavDPCommandSource.initialize() must be called before update().")

        base_state = self._controller.get_base_state()
        robot_pose = tuple(float(v) for v in np.asarray(base_state.position_w, dtype=np.float32).reshape(-1)[:3])
        robot_quat = np.asarray(base_state.quat_wxyz, dtype=np.float32)
        robot_yaw = float(quat_wxyz_to_yaw(robot_quat))
        self._last_robot_pose_xyz = robot_pose

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
                    source="g1_bridge",
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
            self.supervisor.process_frame(batch, publish=self._launch_mode == "g1_view")
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
                metadata={"source": "g1_bridge_pointgoal"},
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
                self._manual_command = self._build_planner_managed_command(task_id="dual", source="g1_bridge_dual")
            return
        if self._mode == "interactive":
            self.planning_session.ensure_navdp_service_ready(context="interactive startup")
            self._manual_command = self._build_planner_managed_command(task_id="interactive", source="g1_bridge_interactive")

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
                cancelled = self.planning_session.cancel_interactive_task()
                if cancelled:
                    self.supervisor.memory_service.clear_planner_task(
                        task_state="cancelled",
                        reason="interactive task cancelled",
                    )
                    print("[G1_INTERACTIVE][TASK] cancel requested")
                else:
                    print("[G1_INTERACTIVE][ROAM] no active task to cancel")
                continue
            if lowered == "/quit":
                self._arm_exit(0, "interactive quit requested")
                return
            if lowered.startswith("/"):
                print(f"[G1_INTERACTIVE][ROAM] unknown command: {text}")
                continue

            try:
                self.planning_session.ensure_navdp_service_ready(context="interactive task")
                self.planning_session.ensure_dual_service_ready(context="interactive task")
                command_id = self.planning_session.submit_interactive_instruction(text)
                self.supervisor.memory_service.set_planner_task(
                    instruction=text,
                    planner_mode="interactive",
                    task_state="pending",
                    task_id="interactive",
                    command_id=int(command_id),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[G1_INTERACTIVE][TASK] rejected instruction={text!r} error={type(exc).__name__}: {exc}")
                continue
            print(f"[G1_INTERACTIVE][TASK] command_id={command_id} queued instruction={text!r}")

    def _ensure_view_runtime(self) -> None:
        if self._launch_mode != "g1_view":
            if self._supervisor is None:
                self._supervisor = Supervisor(config=self._supervisor_config)
            return
        if self._runtime_io is not None:
            return
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
        self._supervisor = Supervisor(
            bus=self._runtime_io.bus,
            shm_ring=self._runtime_io.shm_ring,
            config=self._supervisor_config,
        )
        print(
            "[G1_VIEW] viewer bridge ready "
            f"control={getattr(self.args, 'viewer_control_endpoint', '')} "
            f"telemetry={getattr(self.args, 'viewer_telemetry_endpoint', '')} "
            f"shm={getattr(self.args, 'viewer_shm_name', '')}"
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

    def _sync_planner_scratchpad(self, update: TrajectoryUpdate) -> None:
        if self._mode == "interactive":
            current_phase = str(update.interactive_phase or "")
            current_command_id = int(update.interactive_command_id)
            if current_phase == "task_active" and update.interactive_instruction.strip() != "":
                if self._last_interactive_phase != "task_active" or self._last_interactive_command_id != current_command_id:
                    self.supervisor.memory_service.set_planner_task(
                        instruction=update.interactive_instruction,
                        planner_mode="interactive",
                        task_state="active",
                        task_id="interactive",
                        command_id=current_command_id,
                    )
            elif self._last_interactive_phase == "task_active" and current_phase == "roaming":
                clear_state = "completed" if bool(update.stop) else "idle"
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
    launch_mode = str(getattr(args, "resolved_launch_mode", resolve_launch_mode(args))).strip().lower()
    if bool(args.headless) and launch_mode != "g1_view":
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
        return run_g1_play(args, simulation_app, command_source=NavDPCommandSource(args))
    except Exception as exc:  # noqa: BLE001
        print(f"[G1_POINTGOAL] unhandled exception: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
