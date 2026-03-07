from __future__ import annotations

import threading
import time

import numpy as np
from common.geometry import within_xy_radius, xy_distance
from control.trajectory_tracker import TrajectoryTracker, TrajectoryTrackerConfig

from .g1_bridge_args import apply_demo_defaults, build_arg_parser, validate_args
from .planning import PlannerSession, TrajectoryUpdate


class NavDPCommandSource:
    def __init__(self, args):
        self.args = args
        self.quit_requested = False
        self.exit_code = 0
        self.shutdown_reason = ""

        self._controller = None
        self._planner = PlannerSession(args)
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
        self._pending_exit_code: int | None = None
        self._pending_exit_frames = 0
        self._pending_exit_reason = ""
        self._interactive_running = False
        self._interactive_input_thread: threading.Thread | None = None

    def initialize(self, simulation_app, stage, controller) -> None:
        self._controller = controller
        for _ in range(max(int(self.args.startup_updates), 0)):
            simulation_app.update()
        self._planner.initialize(simulation_app, stage)
        if self._planner.mode == "interactive":
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
        update = self._planner.update(frame_idx)
        now = time.monotonic()

        if update.plan_version > self._last_applied_plan_version:
            self._tracker.set_trajectory(
                update.trajectory_world,
                plan_version=int(update.plan_version),
                timestamp=now,
            )
            self._last_applied_plan_version = int(update.plan_version)

        force_stop = False
        reached_goal = False
        safety_timeout = False
        goal_distance_m = -1.0
        object_distance_m = -1.0
        object_reached = False

        if self._planner.mode == "pointgoal":
            if self._planner.goal_world_xy is not None:
                goal_distance_m = float(np.linalg.norm(self._planner.goal_world_xy - base_state.position_w[:2]))
                if goal_distance_m <= float(self.args.goal_tolerance_m):
                    force_stop = True
                    reached_goal = True
        elif self._planner.mode == "dual":
            if update.stop:
                force_stop = True
                reached_goal = True
            no_response_sec = self._planner.no_response_sec()
            if no_response_sec > float(self.args.safety_timeout_sec):
                force_stop = True
                safety_timeout = True
            if self._planner.demo_object is not None:
                object_distance_m = xy_distance(base_state.position_w, self._planner.demo_object.world_xyz)
                object_reached = within_xy_radius(
                    base_state.position_w,
                    self._planner.demo_object.world_xyz,
                    self._planner.demo_object.stop_radius_m,
                )
                if object_reached:
                    force_stop = True
                    reached_goal = True

        tracker_result = self._tracker.compute_command(
            base_state.position_w,
            base_state.quat_wxyz,
            now=now,
            force_stop=force_stop,
        )
        self._command = tracker_result.command

        if reached_goal:
            if self._planner.mode == "pointgoal":
                self._arm_exit(0, f"goal reached at step={frame_idx} dist={goal_distance_m:.3f}m")
            elif object_reached and self._planner.demo_object is not None:
                self._arm_exit(
                    0,
                    "demo object reached at step="
                    f"{frame_idx} dist={object_distance_m:.3f}m prim={self._planner.demo_object.prim_path}",
                )
            else:
                self._arm_exit(
                    0,
                    f"dual stop signaled at step={frame_idx} goal_v={update.goal_version} traj_v={update.traj_version}",
                )
        elif safety_timeout:
            self._arm_exit(
                1,
                f"safety timeout at step={frame_idx} no_response_sec={no_response_sec:.2f} "
                f"limit={float(self.args.safety_timeout_sec):.2f}",
            )

        if self._pending_exit_code is not None:
            if self._pending_exit_frames <= 0:
                if self._planner.mode == "pointgoal":
                    prefix = "[G1_POINTGOAL]"
                elif self._planner.mode == "interactive":
                    prefix = "[G1_INTERACTIVE]"
                elif self._planner.demo_object is not None:
                    prefix = "[G1_OBJECT_SEARCH]"
                else:
                    prefix = "[G1_DUAL]"
                reason = self._pending_exit_reason if self._pending_exit_reason != "" else "quit requested"
                print(f"{prefix} shutdown reason: {reason}")
                self.quit_requested = True
                self.exit_code = int(self._pending_exit_code)
                self.shutdown_reason = reason
            else:
                self._pending_exit_frames -= 1

        log_interval = max(int(self.args.log_interval), 1)
        if self._planner.mode == "interactive" and update.interactive_phase == "roaming":
            log_interval = max(int(self.args.interactive_idle_log_interval), 1)

        if frame_idx % log_interval == 0:
            self._log_step(frame_idx, update, tracker_result.command, goal_distance_m, object_distance_m)

    def command(self) -> np.ndarray:
        return self._command.copy()

    def shutdown(self) -> None:
        self._interactive_running = False
        self._planner.shutdown()

    def _arm_exit(self, exit_code: int, reason: str) -> None:
        if self._pending_exit_code is None:
            self._pending_exit_code = int(exit_code)
            self._pending_exit_reason = str(reason)
            self._pending_exit_frames = 1

    def _print_interactive_help(self) -> None:
        print("[G1_INTERACTIVE][ROAM] terminal natural-language control")
        print("[G1_INTERACTIVE][ROAM]   text      : submit a new navigation command")
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
                cancelled = self._planner.cancel_interactive_task()
                if cancelled:
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
                command_id = self._planner.submit_interactive_instruction(text)
            except Exception as exc:  # noqa: BLE001
                print(f"[G1_INTERACTIVE][TASK] failed to queue instruction: {type(exc).__name__}: {exc}")
                continue
            print(f"[G1_INTERACTIVE][TASK] command_id={command_id} queued instruction={text!r}")

    def _log_step(
        self,
        frame_idx: int,
        update: TrajectoryUpdate,
        command: np.ndarray,
        goal_distance_m: float,
        object_distance_m: float,
    ) -> None:
        error_note = f" last_error={update.stats.last_error}" if update.stats.last_error != "" else ""
        if self._planner.mode == "pointgoal":
            local_goal = update.goal_local_xy if update.goal_local_xy is not None else np.zeros(2, dtype=np.float32)
            print(
                "[G1_POINTGOAL]"
                f"[step={frame_idx}] goal_dist={goal_distance_m:.3f}m "
                f"goal_local=({float(local_goal[0]):.3f},{float(local_goal[1]):.3f}) "
                f"cmd=({float(command[0]):.3f},{float(command[1]):.3f},{float(command[2]):.3f}) "
                f"plan_ok={update.stats.successful_calls} plan_fail={update.stats.failed_calls} "
                f"plan_latency_ms={update.stats.latency_ms:.1f}{error_note}"
            )
            return

        if self._planner.mode == "interactive":
            if update.interactive_phase == "task_active":
                print(
                    "[G1_INTERACTIVE][TASK]"
                    f"[step={frame_idx}] command_id={update.interactive_command_id} "
                    f"goal_v={update.goal_version} traj_v={update.traj_version} "
                    f"stale_sec={update.stale_sec:.2f} "
                    f"cmd=({float(command[0]):.3f},{float(command[1]):.3f},{float(command[2]):.3f}) "
                    f"plan_ok={update.stats.successful_calls} plan_fail={update.stats.failed_calls} "
                    f"plan_latency_ms={update.stats.latency_ms:.1f}{error_note}"
                )
            else:
                print(
                    "[G1_INTERACTIVE][ROAM]"
                    f"[step={frame_idx}] "
                    f"cmd=({float(command[0]):.3f},{float(command[1]):.3f},{float(command[2]):.3f}) "
                    f"plan_ok={update.stats.successful_calls} plan_fail={update.stats.failed_calls} "
                    f"plan_latency_ms={update.stats.latency_ms:.1f}{error_note}"
                )
            return

        if self._planner.demo_object is not None:
            print(
                "[G1_OBJECT_SEARCH]"
                f"[step={frame_idx}] object_dist={object_distance_m:.3f}m "
                f"goal_v={update.goal_version} traj_v={update.traj_version} "
                f"demo_object={self._planner.demo_object.prim_path} "
                f"stale_sec={update.stale_sec:.2f} "
                f"cmd=({float(command[0]):.3f},{float(command[1]):.3f},{float(command[2]):.3f}) "
                f"plan_ok={update.stats.successful_calls} plan_fail={update.stats.failed_calls} "
                f"plan_latency_ms={update.stats.latency_ms:.1f}{error_note}"
            )
            return

        print(
            "[G1_DUAL]"
            f"[step={frame_idx}] goal_v={update.goal_version} traj_v={update.traj_version} "
            f"stale_sec={update.stale_sec:.2f} "
            f"cmd=({float(command[0]):.3f},{float(command[1]):.3f},{float(command[2]):.3f}) "
            f"plan_ok={update.stats.successful_calls} plan_fail={update.stats.failed_calls} "
            f"plan_latency_ms={update.stats.latency_ms:.1f}{error_note}"
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
