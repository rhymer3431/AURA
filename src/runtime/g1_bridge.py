from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from common.geometry import quat_wxyz_to_yaw, within_xy_radius, wrap_to_pi, xy_distance
from control.trajectory_tracker import TrajectoryTracker, TrajectoryTrackerConfig
from ipc.messages import ActionCommand, ActionStatus, FrameHeader

from .g1_bridge_args import apply_demo_defaults, build_arg_parser, validate_args
from .planning_session import PlannerStats, PlanningSession, TrajectoryUpdate
from .supervisor import Supervisor, SupervisorConfig


@dataclass(frozen=True)
class CommandEvaluation:
    force_stop: bool
    goal_distance_m: float
    yaw_error_rad: float
    reached_goal: bool


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

        self._controller = None
        self._mode = str(getattr(args, "planner_mode", "interactive")).strip().lower()
        self._planning_session = planning_session or PlanningSession(args)
        self._supervisor = supervisor or Supervisor(
            config=SupervisorConfig(
                memory_db_path=str(getattr(args, "memory_db_path", "state/memory/memory.sqlite")),
                detector_engine_path=str(getattr(args, "detector_engine_path", "")),
            )
        )
        self._tracker = TrajectoryTracker(
            TrajectoryTrackerConfig(
                max_vx=float(args.cmd_max_vx),
                max_vy=float(args.cmd_max_vy),
                max_wz=float(args.cmd_max_wz),
                lookahead_distance_m=float(args.lookahead_distance_m),
                heading_slowdown_rad=float(args.heading_slowdown_rad),
                traj_stale_timeout_sec=float(args.traj_stale_timeout_sec),
                cmd_accel_limit=float(args.cmd_accel_limit),
                cmd_yaw_accel_limit=float(args.cmd_yaw_accel_limit),
            )
        )
        self._command = np.zeros(3, dtype=np.float32)
        self._last_applied_plan_version = -1
        self._active_command: ActionCommand | None = None
        self._pending_status: ActionStatus | None = None
        self._manual_command: ActionCommand | None = None
        self._last_robot_pose_xyz = (0.0, 0.0, 0.0)
        self._pending_exit_code: int | None = None
        self._pending_exit_frames = 0
        self._pending_exit_reason = ""
        self._interactive_running = False
        self._interactive_input_thread: threading.Thread | None = None

    @property
    def supervisor(self) -> Supervisor:
        return self._supervisor

    @property
    def planning_session(self) -> PlanningSession:
        return self._planning_session

    def initialize(self, simulation_app, stage, controller) -> None:
        self._controller = controller
        for _ in range(max(int(self.args.startup_updates), 0)):
            simulation_app.update()
        self._planning_session.initialize(simulation_app, stage)
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

        observation = self._planning_session.capture_observation(frame_idx)
        if observation is not None:
            batch = IsaacObservationBatch(
                frame_header=FrameHeader(
                    frame_id=int(observation.frame_id),
                    timestamp_ns=time.time_ns(),
                    source="g1_bridge",
                    width=int(observation.rgb.shape[1]),
                    height=int(observation.rgb.shape[0]),
                    camera_pose_xyz=tuple(float(v) for v in observation.cam_pos[:3]),
                    camera_quat_wxyz=tuple(float(v) for v in observation.cam_quat[:4]),
                    metadata=dict(observation.sensor_meta),
                ),
                robot_pose_xyz=robot_pose,
                rgb_image=observation.rgb,
                depth_image_m=observation.depth,
                camera_intrinsic=observation.intrinsic,
            )
            self._supervisor.process_frame(batch, publish=False)

        command = self._resolve_action_command(robot_pose=robot_pose)
        update = self._plan_command(
            frame_idx=frame_idx,
            observation=observation,
            command=command,
            robot_pos_world=np.asarray(base_state.position_w, dtype=np.float32),
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat,
        )
        evaluation = self._evaluate_command(command, base_state.position_w, robot_yaw)

        now = time.monotonic()
        if update.plan_version > self._last_applied_plan_version:
            self._tracker.set_trajectory(update.trajectory_world, plan_version=int(update.plan_version), timestamp=now)
            self._last_applied_plan_version = int(update.plan_version)

        if command is not None and command.action_type == "LOOK_AT":
            self._command = self._look_at_command(command, robot_yaw)
        else:
            tracker_result = self._tracker.compute_command(
                np.asarray(base_state.position_w, dtype=np.float32),
                robot_quat,
                now=now,
                force_stop=evaluation.force_stop,
            )
            self._command = tracker_result.command

        self._pending_status = self._build_status(
            command=command,
            update=update,
            robot_pose=robot_pose,
            evaluation=evaluation,
        )

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
        if self._mode == "interactive" and command is None:
            log_interval = max(int(self.args.interactive_idle_log_interval), 1)
        if frame_idx % log_interval == 0:
            self._log_step(frame_idx, update, command, evaluation)

    def command(self) -> np.ndarray:
        return self._command.copy()

    def shutdown(self) -> None:
        self._interactive_running = False
        self._planning_session.shutdown()

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
                target_json: dict[str, object] = {}
                if bool(getattr(self.args, "spawn_demo_object", False)):
                    target_json["target_class"] = "cube"
                self._supervisor.submit_task(instruction, target_json=target_json)

    def _resolve_action_command(self, *, robot_pose: tuple[float, float, float]) -> ActionCommand | None:
        if self._manual_command is not None:
            self._active_command = self._manual_command
            return self._manual_command
        command = self._supervisor.step(
            now=time.time(),
            robot_pose=robot_pose,
            action_status=self._pending_status,
            publish=False,
        )
        self._pending_status = None
        self._active_command = command
        return command

    def _plan_command(
        self,
        *,
        frame_idx: int,
        observation,
        command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> TrajectoryUpdate:
        if observation is None:
            return self._empty_update(frame_idx=frame_idx, action_command=command, error="sensor data unavailable", stop=True)
        if command is not None and command.action_type == "LOOK_AT":
            return self._empty_update(frame_idx=frame_idx, action_command=command, stop=False)
        return self._planning_session.plan_with_observation(
            observation,
            action_command=command,
            robot_pos_world=robot_pos_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
        )

    def _evaluate_command(
        self,
        command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> CommandEvaluation:
        if command is None:
            return CommandEvaluation(force_stop=True, goal_distance_m=-1.0, yaw_error_rad=0.0, reached_goal=False)
        if command.action_type == "STOP":
            return CommandEvaluation(force_stop=True, goal_distance_m=0.0, yaw_error_rad=0.0, reached_goal=False)
        if command.action_type == "LOOK_AT":
            yaw_target = float(command.look_at_yaw_rad if command.look_at_yaw_rad is not None else robot_yaw)
            yaw_error = wrap_to_pi(yaw_target - robot_yaw)
            return CommandEvaluation(
                force_stop=abs(float(yaw_error)) < 0.05,
                goal_distance_m=0.0,
                yaw_error_rad=float(yaw_error),
                reached_goal=False,
            )
        if command.target_pose_xyz is None:
            return CommandEvaluation(force_stop=False, goal_distance_m=-1.0, yaw_error_rad=0.0, reached_goal=False)
        reached_goal = False
        goal_distance = xy_distance(np.asarray(robot_pos_world, dtype=np.float32), np.asarray(command.target_pose_xyz, dtype=np.float32))
        if command.action_type in {"NAV_TO_POSE", "NAV_TO_PLACE"}:
            reached_goal = within_xy_radius(
                np.asarray(robot_pos_world, dtype=np.float32),
                np.asarray(command.target_pose_xyz, dtype=np.float32),
                float(command.stop_radius_m),
            )
        return CommandEvaluation(
            force_stop=bool(reached_goal),
            goal_distance_m=float(goal_distance),
            yaw_error_rad=0.0,
            reached_goal=bool(reached_goal),
        )

    def _build_status(
        self,
        *,
        command: ActionCommand | None,
        update: TrajectoryUpdate,
        robot_pose: tuple[float, float, float],
        evaluation: CommandEvaluation,
    ) -> ActionStatus | None:
        if command is None:
            return None
        if evaluation.reached_goal:
            return ActionStatus(
                command_id=command.command_id,
                state="succeeded",
                success=True,
                robot_pose_xyz=robot_pose,
                distance_remaining_m=max(float(evaluation.goal_distance_m), 0.0),
                metadata={"action_type": command.action_type},
            )
        if update.stats.last_error != "" and update.trajectory_world.shape[0] == 0 and command.action_type not in {"STOP", "LOOK_AT"}:
            return ActionStatus(
                command_id=command.command_id,
                state="failed",
                success=False,
                reason=update.stats.last_error,
                robot_pose_xyz=robot_pose,
                distance_remaining_m=None if evaluation.goal_distance_m < 0.0 else float(evaluation.goal_distance_m),
                metadata={"action_type": command.action_type},
            )
        if command.action_type == "LOOK_AT" and abs(float(evaluation.yaw_error_rad)) < 0.05:
            return ActionStatus(
                command_id=command.command_id,
                state="succeeded",
                success=True,
                robot_pose_xyz=robot_pose,
                distance_remaining_m=0.0,
                metadata={"action_type": command.action_type},
            )
        return ActionStatus(
            command_id=command.command_id,
            state="running",
            success=False,
            robot_pose_xyz=robot_pose,
            distance_remaining_m=None if evaluation.goal_distance_m < 0.0 else float(evaluation.goal_distance_m),
            metadata={"action_type": command.action_type},
        )

    def _look_at_command(self, command: ActionCommand, robot_yaw: float) -> np.ndarray:
        yaw_target = float(command.look_at_yaw_rad if command.look_at_yaw_rad is not None else robot_yaw)
        yaw_error = wrap_to_pi(yaw_target - robot_yaw)
        wz = np.clip(1.5 * float(yaw_error), -float(self.args.cmd_max_wz), float(self.args.cmd_max_wz))
        return np.asarray([0.0, 0.0, float(wz)], dtype=np.float32)

    def _empty_update(
        self,
        *,
        frame_idx: int,
        action_command: ActionCommand | None,
        error: str = "",
        stop: bool = False,
    ) -> TrajectoryUpdate:
        failed_calls = 1 if error != "" else 0
        return TrajectoryUpdate(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=self._last_applied_plan_version,
            stats=PlannerStats(failed_calls=failed_calls, last_error=error, last_plan_step=int(frame_idx)),
            source_frame_id=int(frame_idx),
            action_command=action_command,
            stop=bool(stop),
        )

    def _arm_exit(self, exit_code: int, reason: str) -> None:
        if self._pending_exit_code is None:
            self._pending_exit_code = int(exit_code)
            self._pending_exit_reason = str(reason)
            self._pending_exit_frames = 1

    def _print_interactive_help(self) -> None:
        print("[G1_INTERACTIVE][ROAM] terminal natural-language control")
        print("[G1_INTERACTIVE][ROAM]   text      : submit a new direct task request")
        print("[G1_INTERACTIVE][ROAM]   /help     : show this help")
        print("[G1_INTERACTIVE][ROAM]   /cancel   : cancel the active task and stop")
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
                self._supervisor.orchestrator.cancel_active_task(reason="interactive_cancel")
                print("[G1_INTERACTIVE][TASK] cancel requested")
                continue
            if lowered == "/quit":
                self._arm_exit(0, "interactive quit requested")
                return
            if lowered.startswith("/"):
                print(f"[G1_INTERACTIVE][ROAM] unknown command: {text}")
                continue

            request = self._supervisor.submit_task(text)
            print(f"[G1_INTERACTIVE][TASK] task_id={request.task_id} queued instruction={text!r}")

    def _log_prefix(self) -> str:
        if self._mode == "pointgoal":
            return "[G1_POINTGOAL]"
        if self._mode == "interactive":
            return "[G1_INTERACTIVE]"
        return "[G1_DIRECT]"

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


def main() -> int:
    try:
        args = build_arg_parser().parse_args()
        args = apply_demo_defaults(args)
        validate_args(args)
    except ValueError as exc:
        print(f"[G1_POINTGOAL] {exc}")
        return 2

    from isaacsim import SimulationApp
    from locomotion.runtime import run as run_g1_play

    launch_config = {"headless": bool(args.headless)}
    if bool(args.headless):
        launch_config["disable_viewport_updates"] = True
    simulation_app = SimulationApp(launch_config=launch_config)

    try:
        return run_g1_play(args, simulation_app, command_source=NavDPCommandSource(args))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
