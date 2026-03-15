from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import requests

from adapters.dual_http import DualSystemClient, DualSystemClientConfig
from adapters.sensors.d455_sensor import D455SensorAdapter, D455SensorAdapterConfig
from common.geometry import world_goal_to_robot_frame
from control.async_planners import (
    AsyncDualPlanner,
    AsyncNoGoalPlanner,
    AsyncPointGoalPlanner,
    DualPlannerInput,
    NoGoalPlannerInput,
    PlannerInput,
    PlannerOutput,
)
from inference.navdp import InProcessNavDPClient, create_inprocess_navdp_client
from ipc.messages import ActionCommand
from memory.models import MemoryContextBundle


def _health_url(base_url: str) -> str:
    return f"{str(base_url).rstrip('/')}/health"


def _check_remote_service(base_url: str, *, timeout_sec: float, service_name: str, context: str) -> None:
    url = str(base_url).strip()
    if url == "":
        raise RuntimeError(f"{context}: missing {service_name} base URL")
    try:
        response = requests.get(_health_url(url), timeout=float(timeout_sec))
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{context}: {service_name} is unavailable at {url}. "
            f"Start the required server first. detail={type(exc).__name__}: {exc}"
        ) from exc


@dataclass(frozen=True)
class PlannerStats:
    successful_calls: int = 0
    failed_calls: int = 0
    latency_ms: float = 0.0
    last_error: str = ""
    last_plan_step: int = -1


@dataclass(frozen=True)
class ExecutionObservation:
    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    sensor_meta: dict[str, Any]
    cam_pos: np.ndarray
    cam_quat: np.ndarray
    intrinsic: np.ndarray
    memory_context: MemoryContextBundle | None = None


@dataclass(frozen=True)
class TrajectoryUpdate:
    trajectory_world: np.ndarray
    plan_version: int
    stats: PlannerStats
    source_frame_id: int
    goal_local_xy: np.ndarray | None = None
    action_command: ActionCommand | None = None
    stop: bool = False
    planner_control_mode: str | None = None
    planner_yaw_delta_rad: float | None = None
    stale_sec: float = -1.0
    goal_version: int = -1
    traj_version: int = -1
    used_cached_traj: bool = False
    sensor_meta: dict[str, Any] | None = None
    interactive_phase: str | None = None
    interactive_command_id: int = -1
    interactive_instruction: str = ""


class PlanningSession:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        sensor_factory: Callable[[D455SensorAdapterConfig], D455SensorAdapter] | None = None,
        navdp_client_factory: Callable[[np.ndarray, argparse.Namespace], Any] | None = None,
        dual_client_factory: Callable[[argparse.Namespace], DualSystemClient] | None = None,
    ) -> None:
        self.args = args
        self.mode = str(getattr(args, "planner_mode", "pointgoal")).strip().lower()
        self.sensor_factory = sensor_factory or (lambda cfg: D455SensorAdapter(cfg))
        self.navdp_client_factory = navdp_client_factory or self._default_navdp_client_factory
        self.dual_client_factory = dual_client_factory or self._default_dual_client_factory
        self.sensor: D455SensorAdapter | None = None
        self.navdp_client: Any | None = None
        self.pointgoal_planner: AsyncPointGoalPlanner | None = None
        self.nogoal_planner: AsyncNoGoalPlanner | None = None
        self.dual_planner: AsyncDualPlanner | None = None
        self._dual_client: DualSystemClient | None = None
        self._intrinsic = np.eye(3, dtype=np.float32)
        self._dual_instruction = ""

        self._last_plan_version = -1
        self._last_trajectory = np.zeros((0, 3), dtype=np.float32)
        self._last_goal_local_xy = np.zeros(2, dtype=np.float32)
        self._last_planner_control_mode: str | None = None
        self._last_planner_yaw_delta_rad: float | None = None
        self._last_planner_control_reason = ""
        self._last_used_cached_traj = False
        self._last_server_stale_sec = -1.0
        self._last_goal_version = -1
        self._last_traj_version = -1
        self._last_system2_pixel_goal: list[int] | None = None
        self._last_dual_response_ts = time.perf_counter()
        self._stats = PlannerStats()
        self.last_sensor_init_report: dict[str, Any] = {}

        self._interactive_lock = threading.Lock()
        self._interactive_state = "roaming" if self.mode == "interactive" else ""
        self._interactive_command_seq = 0
        self._interactive_pending_command_id = -1
        self._interactive_pending_instruction = ""
        self._interactive_cancel_requested = False
        self._interactive_active_command_id = -1
        self._interactive_active_instruction = ""
        self._interactive_session_plan_version = -1
        self._interactive_last_nogoal_plan_version = -1
        self._interactive_last_dual_plan_version = -1
        self._interactive_last_nogoal_failed_calls = 0
        self._interactive_last_dual_failed_calls = 0

    @property
    def navdp_backend_name(self) -> str:
        if self.navdp_client is None:
            return ""
        return str(getattr(self.navdp_client, "backend_name", getattr(self.navdp_client, "backend_impl", "")))

    @property
    def stats(self) -> PlannerStats:
        return self._stats

    def active_memory_instruction(self) -> str:
        if self.mode == "dual":
            return str(self._dual_instruction).strip()
        if self.mode == "interactive" and self._interactive_state == "task_active":
            return str(self._interactive_active_instruction).strip()
        return ""

    def initialize(self, simulation_app, stage) -> None:
        self.sensor = self.sensor_factory(
            D455SensorAdapterConfig(
                use_d455=True,
                image_width=int(getattr(self.args, "image_width", 640)),
                image_height=int(getattr(self.args, "image_height", 640)),
                depth_max_m=float(getattr(self.args, "depth_max_m", 5.0)),
                strict_d455=bool(getattr(self.args, "strict_d455", False)),
                force_runtime_mount=bool(getattr(self.args, "force_runtime_camera", False)),
            )
        )
        init_ok, init_msg = self.sensor.initialize(simulation_app, stage)
        self.last_sensor_init_report = {
            "ok": bool(init_ok),
            "message": str(init_msg),
            "capture_report": dict(self.sensor.last_capture_meta),
            "camera_prim_path": str(self.sensor.rgb_prim_path or ""),
            "depth_camera_prim_path": str(self.sensor.depth_prim_path or ""),
            "runtime_mount": bool(self.sensor.runtime_camera_mode),
        }
        print(f"[PLANNING_SESSION] sensor init: ok={init_ok} msg={init_msg}")
        if not init_ok:
            raise RuntimeError(init_msg)
        self.initialize_local(intrinsic=self.sensor.intrinsic)

    def initialize_local(
        self,
        *,
        intrinsic: np.ndarray,
        navdp_client: Any | None = None,
        dual_client: DualSystemClient | None = None,
    ) -> None:
        self._intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
        self.navdp_client = navdp_client or self.navdp_client_factory(self._intrinsic.copy(), self.args)
        self.navdp_client.navigator_reset(self._intrinsic.copy(), batch_size=1)
        self.pointgoal_planner = AsyncPointGoalPlanner(
            client=self.navdp_client,
            use_trajectory_z=bool(getattr(self.args, "use_trajectory_z", False)),
            pointgoal_frame="robot",
        )
        self.nogoal_planner = AsyncNoGoalPlanner(
            client=self.navdp_client,
            use_trajectory_z=bool(getattr(self.args, "use_trajectory_z", False)),
        )
        self.pointgoal_planner.start()
        self.nogoal_planner.start()
        if self.mode in {"dual", "interactive"}:
            self._dual_client = dual_client or self.dual_client_factory(self.args)
            self.dual_planner = AsyncDualPlanner(client=self._dual_client)
            self.dual_planner.start()
        if self.mode == "interactive":
            if not self._activate_roaming("startup"):
                raise RuntimeError(self._stats.last_error or "interactive roaming initialization failed")

    def shutdown(self) -> None:
        for planner in (self.pointgoal_planner, self.nogoal_planner, self.dual_planner):
            if planner is not None:
                planner.stop()

    def ensure_navdp_service_ready(self, *, context: str) -> None:
        try:
            _check_remote_service(
                str(getattr(self.args, "server_url", "")),
                timeout_sec=float(getattr(self.args, "timeout_sec", 5.0)),
                service_name="NavDP server",
                context=context,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} Suggested command: .\\run_navdp_server.ps1") from exc

    def ensure_dual_service_ready(self, *, context: str) -> None:
        try:
            _check_remote_service(
                str(getattr(self.args, "dual_server_url", "")),
                timeout_sec=float(getattr(self.args, "timeout_sec", 5.0)),
                service_name="dual server",
                context=context,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} Suggested command: .\\run_vlm_dual_server.ps1") from exc

    def start_dual_task(self, instruction: str) -> None:
        if self.mode != "dual":
            raise RuntimeError("start_dual_task requires planner-mode=dual")
        text = str(instruction).strip()
        if text == "":
            raise ValueError("dual instruction must be non-empty")
        if self._dual_client is None or self.dual_planner is None:
            raise RuntimeError("dual planner is not initialized")
        self._dual_instruction = text
        self._dual_reset(text, prefix="[G1_DUAL]")
        self.dual_planner.reset_state()
        self._last_plan_version = -1
        self._last_trajectory = np.zeros((0, 3), dtype=np.float32)
        self._reset_planner_control_state()
        self._last_server_stale_sec = -1.0
        self._last_goal_version = -1
        self._last_traj_version = -1
        self._last_system2_pixel_goal = None
        self._last_dual_response_ts = time.perf_counter()
        self._stats = PlannerStats()

    def submit_interactive_instruction(self, instruction: str) -> int:
        if self.mode != "interactive":
            raise RuntimeError("submit_interactive_instruction requires planner-mode=interactive")
        text = str(instruction).strip()
        if text == "":
            raise ValueError("interactive instruction must be non-empty")
        with self._interactive_lock:
            self._interactive_command_seq += 1
            self._interactive_pending_command_id = int(self._interactive_command_seq)
            self._interactive_pending_instruction = text
            self._interactive_cancel_requested = False
            return int(self._interactive_pending_command_id)

    def cancel_interactive_task(self) -> bool:
        if self.mode != "interactive":
            return False
        with self._interactive_lock:
            has_work = (
                self._interactive_pending_command_id >= 0
                or self._interactive_active_command_id >= 0
                or self._interactive_state in {"task_pending", "task_active"}
            )
            if not has_work:
                return False
            self._interactive_pending_command_id = -1
            self._interactive_pending_instruction = ""
            self._interactive_cancel_requested = True
            return True

    def capture_observation(self, frame_id: int, *, env=None) -> ExecutionObservation | None:  # noqa: ANN001
        if self.sensor is None:
            raise RuntimeError("PlanningSession.initialize() must be called first.")
        rgb, depth, sensor_meta = self.sensor.capture_rgbd_with_meta(env)
        if rgb is None or depth is None:
            return None
        cam_pos, cam_quat = self.sensor.get_rgb_camera_pose_world()
        if cam_pos is None:
            cam_pos = np.zeros(3, dtype=np.float32)
        if cam_quat is None:
            cam_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return ExecutionObservation(
            frame_id=int(frame_id),
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            sensor_meta=dict(sensor_meta),
            cam_pos=np.asarray(cam_pos, dtype=np.float32),
            cam_quat=np.asarray(cam_quat, dtype=np.float32),
            intrinsic=self._intrinsic.copy(),
        )

    def update(
        self,
        frame_id: int,
        *,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
        env=None,  # noqa: ANN001
    ) -> TrajectoryUpdate:
        observation = self.capture_observation(frame_id, env=env)
        if observation is None:
            return TrajectoryUpdate(
                trajectory_world=self._last_trajectory.copy(),
                plan_version=self._last_plan_version,
                stats=PlannerStats(failed_calls=1, last_error="sensor data unavailable", last_plan_step=int(frame_id)),
                source_frame_id=int(frame_id),
                action_command=action_command,
                stop=True,
            )
        return self.plan_with_observation(
            observation,
            action_command=action_command,
            robot_pos_world=robot_pos_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
        )

    def plan_with_observation(
        self,
        observation: ExecutionObservation,
        *,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> TrajectoryUpdate:
        del robot_quat_wxyz
        if self.mode == "dual":
            if self.dual_planner is None:
                raise RuntimeError("PlanningSession is not initialized.")
            return self._run_dual(observation, action_command=action_command)
        if self.mode == "interactive":
            if self.nogoal_planner is None or self.dual_planner is None:
                raise RuntimeError("PlanningSession is not initialized.")
            return self._run_interactive(observation, action_command=action_command)
        if self.pointgoal_planner is None or self.nogoal_planner is None:
            raise RuntimeError("PlanningSession is not initialized.")
        if action_command is None or action_command.action_type in {"STOP", "LOOK_AT"}:
            return TrajectoryUpdate(
                trajectory_world=np.zeros((0, 3), dtype=np.float32),
                plan_version=self._last_plan_version,
                stats=PlannerStats(last_plan_step=int(observation.frame_id)),
                source_frame_id=int(observation.frame_id),
                action_command=action_command,
                stop=True,
                sensor_meta=dict(observation.sensor_meta),
            )
        if action_command.action_type == "LOCAL_SEARCH":
            return self._run_nogoal(observation, action_command=action_command)
        if action_command.target_pose_xyz is None:
            return TrajectoryUpdate(
                trajectory_world=self._last_trajectory.copy(),
                plan_version=self._last_plan_version,
                stats=PlannerStats(
                    failed_calls=1,
                    last_error="target_pose_xyz is required",
                    last_plan_step=int(observation.frame_id),
                ),
                source_frame_id=int(observation.frame_id),
                action_command=action_command,
                stop=True,
                sensor_meta=dict(observation.sensor_meta),
            )
        goal_local_xy = world_goal_to_robot_frame(
            goal_xy=np.asarray(action_command.target_pose_xyz[:2], dtype=np.float32),
            robot_xy=np.asarray(robot_pos_world[:2], dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )
        return self._run_pointgoal(
            observation,
            goal_local_xy,
            robot_pos_world,
            robot_yaw,
            action_command,
        )

    def build_local_observation(
        self,
        *,
        frame_id: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_pose_xyz: tuple[float, float, float] | np.ndarray,
        camera_quat_wxyz: tuple[float, float, float, float] | np.ndarray,
        intrinsic: np.ndarray | None = None,
        sensor_meta: dict[str, Any] | None = None,
        memory_context: MemoryContextBundle | None = None,
    ) -> ExecutionObservation:
        if intrinsic is None:
            intrinsic = self._default_intrinsic(rgb.shape[1], rgb.shape[0])
        return ExecutionObservation(
            frame_id=int(frame_id),
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            sensor_meta=dict(sensor_meta or {}),
            cam_pos=np.asarray(camera_pose_xyz, dtype=np.float32),
            cam_quat=np.asarray(camera_quat_wxyz, dtype=np.float32),
            intrinsic=np.asarray(intrinsic, dtype=np.float32),
            memory_context=memory_context,
        )

    def _run_pointgoal(
        self,
        observation: ExecutionObservation,
        goal_local_xy: np.ndarray,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        action_command: ActionCommand,
    ) -> TrajectoryUpdate:
        assert self.pointgoal_planner is not None
        self.pointgoal_planner.submit(
            PlannerInput(
                frame_id=observation.frame_id,
                local_goal_xy=np.asarray(goal_local_xy, dtype=np.float32),
                rgb=observation.rgb,
                depth=observation.depth,
                sensor_meta=observation.sensor_meta,
                cam_pos=observation.cam_pos,
                cam_quat=observation.cam_quat,
                robot_pos=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(robot_yaw),
            )
        )
        return self._consume_planner_latest(
            planner=self.pointgoal_planner,
            frame_id=observation.frame_id,
            action_command=action_command,
            goal_local_xy=np.asarray(goal_local_xy, dtype=np.float32),
            sensor_meta=observation.sensor_meta,
        )

    def _run_nogoal(self, observation: ExecutionObservation, *, action_command: ActionCommand | None) -> TrajectoryUpdate:
        assert self.nogoal_planner is not None
        self.nogoal_planner.submit(
            NoGoalPlannerInput(
                frame_id=observation.frame_id,
                rgb=observation.rgb,
                depth=observation.depth,
                sensor_meta=observation.sensor_meta,
                cam_pos=observation.cam_pos,
                cam_quat=observation.cam_quat,
            )
        )
        return self._consume_planner_latest(
            planner=self.nogoal_planner,
            frame_id=observation.frame_id,
            action_command=action_command,
            sensor_meta=observation.sensor_meta,
        )

    def _run_dual(self, observation: ExecutionObservation, *, action_command: ActionCommand | None) -> TrajectoryUpdate:
        if self.dual_planner is None:
            raise RuntimeError("dual planner is not initialized")
        if self._dual_instruction == "":
            return TrajectoryUpdate(
                trajectory_world=np.zeros((0, 3), dtype=np.float32),
                plan_version=self._last_plan_version,
                stats=PlannerStats(failed_calls=1, last_error="dual task not started", last_plan_step=int(observation.frame_id)),
                source_frame_id=int(observation.frame_id),
                action_command=action_command,
                stop=True,
                sensor_meta=dict(observation.sensor_meta),
            )
        force_s2 = bool(self._last_trajectory.shape[0] == 0 and self._last_planner_control_mode != "yaw_delta")
        should_plan = self._last_trajectory.shape[0] == 0 or (
            observation.frame_id % max(int(getattr(self.args, "dual_request_gap_frames", 1)), 1) == 0
        )
        if should_plan:
            self.dual_planner.submit(
                DualPlannerInput(
                    frame_id=int(observation.frame_id),
                    rgb=observation.rgb,
                    depth=observation.depth,
                    sensor_meta=dict(observation.sensor_meta),
                    cam_pos=observation.cam_pos,
                    cam_quat=observation.cam_quat,
                    memory_context=observation.memory_context,
                    events={
                        "force_s2": force_s2,
                        "stuck": False,
                        "collision_risk": False,
                    },
                )
            )

        plan = self.dual_planner.consume_latest(last_seen_version=self._last_plan_version)
        plan_stop = bool(self._last_planner_control_mode == "stop")
        if plan is not None:
            self._last_dual_response_ts = time.perf_counter()
            self._accept_dual_plan(plan)
            plan_stop = bool(self._last_planner_control_mode == "stop")

        success_calls, failed_calls, last_error, planner_latency_ms = self.dual_planner.snapshot_status()
        self._stats = PlannerStats(
            successful_calls=int(success_calls),
            failed_calls=int(failed_calls),
            latency_ms=float(planner_latency_ms),
            last_error=str(last_error),
            last_plan_step=int(self._stats.last_plan_step),
        )
        return self._build_update(
            frame_id=observation.frame_id,
            action_command=action_command,
            stop=plan_stop,
            sensor_meta=observation.sensor_meta,
        )

    def _run_interactive(self, observation: ExecutionObservation, *, action_command: ActionCommand | None) -> TrajectoryUpdate:
        self._consume_interactive_controls()
        plan_stop = False
        if self._interactive_state == "roaming":
            self._update_interactive_roaming(observation)
        else:
            plan_stop = self._update_interactive_task(observation)
        return self._build_update(
            frame_id=observation.frame_id,
            action_command=action_command,
            stop=plan_stop,
            sensor_meta=observation.sensor_meta,
        )

    def _consume_interactive_controls(self) -> None:
        if self.mode != "interactive":
            return
        cancel_requested = False
        pending_command_id = -1
        pending_instruction = ""
        with self._interactive_lock:
            cancel_requested = bool(self._interactive_cancel_requested)
            pending_command_id = int(self._interactive_pending_command_id)
            pending_instruction = str(self._interactive_pending_instruction)
            self._interactive_cancel_requested = False
            self._interactive_pending_command_id = -1
            self._interactive_pending_instruction = ""

        if cancel_requested:
            self._activate_roaming("user cancel")
        if pending_command_id >= 0 and pending_instruction != "":
            self._activate_task(pending_command_id, pending_instruction)

    def _activate_roaming(self, reason: str) -> bool:
        if self.navdp_client is None or self.nogoal_planner is None:
            return False
        try:
            algo = self.navdp_client.navigator_reset(self._intrinsic.copy(), batch_size=1)
        except Exception as exc:  # noqa: BLE001
            error = f"roaming navigator_reset failed: {type(exc).__name__}: {exc}"
            self._interactive_state = "roaming"
            self._last_server_stale_sec = -1.0
            self._last_goal_version = -1
            self._last_traj_version = -1
            self._last_system2_pixel_goal = None
            self._stats = PlannerStats(
                successful_calls=0,
                failed_calls=1,
                latency_ms=0.0,
                last_error=error,
                last_plan_step=int(self._stats.last_plan_step),
            )
            self._interactive_log("ROAM", f"{reason} -> {error}")
            self._interactive_clear_active_task()
            self._reset_planner_control_state()
            self._emit_interactive_trajectory(np.zeros((0, 3), dtype=np.float32))
            return False

        self.nogoal_planner.reset_state()
        self._interactive_last_nogoal_plan_version = -1
        self._interactive_last_nogoal_failed_calls = 0
        if self.dual_planner is not None:
            self.dual_planner.reset_state()
        self._interactive_last_dual_plan_version = -1
        self._interactive_last_dual_failed_calls = 0
        self._interactive_clear_active_task()
        self._reset_planner_control_state()
        self._interactive_state = "roaming"
        self._last_server_stale_sec = -1.0
        self._last_goal_version = -1
        self._last_traj_version = -1
        self._last_system2_pixel_goal = None
        self._last_dual_response_ts = time.perf_counter()
        self._stats = PlannerStats()
        self._emit_interactive_trajectory(np.zeros((0, 3), dtype=np.float32))
        self._interactive_log("ROAM", f"state=roaming reason={reason} navdp_reset={algo}")
        return True

    def _activate_task(self, command_id: int, instruction: str) -> bool:
        if self._dual_client is None or self.dual_planner is None:
            return False
        self._interactive_state = "task_pending"
        self._interactive_log(
            "TASK",
            f"state=task_pending command_id={int(command_id)} instruction={instruction!r}",
        )
        self._emit_interactive_trajectory(np.zeros((0, 3), dtype=np.float32))
        if self.nogoal_planner is not None:
            self.nogoal_planner.reset_state()
        self._interactive_last_nogoal_plan_version = -1
        self._interactive_last_nogoal_failed_calls = 0
        self.dual_planner.reset_state()
        self._interactive_last_dual_plan_version = -1
        self._interactive_last_dual_failed_calls = 0
        self._reset_planner_control_state()

        try:
            self._dual_reset(instruction, prefix="[G1_INTERACTIVE][TASK]")
        except Exception as exc:  # noqa: BLE001
            error = f"dual_reset failed for command_id={int(command_id)}: {type(exc).__name__}: {exc}"
            self._stats = PlannerStats(
                successful_calls=0,
                failed_calls=1,
                latency_ms=0.0,
                last_error=error,
                last_plan_step=int(self._stats.last_plan_step),
            )
            self._interactive_log("TASK", error)
            self._activate_roaming(f"dual reset failure for command_id={int(command_id)}")
            return False

        self._interactive_state = "task_active"
        self._interactive_active_command_id = int(command_id)
        self._interactive_active_instruction = str(instruction)
        self._last_server_stale_sec = -1.0
        self._last_goal_version = -1
        self._last_traj_version = -1
        self._last_system2_pixel_goal = None
        self._last_dual_response_ts = time.perf_counter()
        self._stats = PlannerStats()
        self._interactive_log(
            "TASK",
            f"state=task_active command_id={int(command_id)} instruction={instruction!r}",
        )
        return True

    def _update_interactive_roaming(self, observation: ExecutionObservation) -> None:
        if self.nogoal_planner is None:
            raise RuntimeError("interactive no-goal planner is not initialized")
        should_plan = self._last_trajectory.shape[0] == 0 or (
            observation.frame_id % max(int(getattr(self.args, "plan_interval_frames", 1)), 1) == 0
        )
        if should_plan:
            self.nogoal_planner.submit(
                NoGoalPlannerInput(
                    frame_id=int(observation.frame_id),
                    rgb=observation.rgb,
                    depth=observation.depth,
                    sensor_meta=dict(observation.sensor_meta),
                    cam_pos=observation.cam_pos,
                    cam_quat=observation.cam_quat,
                )
            )

        plan = self.nogoal_planner.consume_latest(last_seen_version=self._interactive_last_nogoal_plan_version)
        if plan is not None:
            self._interactive_last_nogoal_plan_version = int(plan.plan_version)
            self._emit_interactive_trajectory(plan.trajectory_world)
            self._stats = PlannerStats(
                successful_calls=int(plan.successful_calls),
                failed_calls=int(plan.failed_calls),
                latency_ms=float(plan.latency_ms),
                last_error=str(plan.last_error),
                last_plan_step=int(plan.source_frame_id),
            )

        success_calls, failed_calls, last_error, planner_latency_ms = self.nogoal_planner.snapshot_status()
        self._interactive_last_nogoal_failed_calls = int(failed_calls)
        self._stats = PlannerStats(
            successful_calls=int(success_calls),
            failed_calls=int(failed_calls),
            latency_ms=float(planner_latency_ms),
            last_error=str(last_error),
            last_plan_step=int(self._stats.last_plan_step),
        )

    def _update_interactive_task(self, observation: ExecutionObservation) -> bool:
        if self.dual_planner is None:
            raise RuntimeError("interactive dual planner is not initialized")
        force_s2 = bool(self._last_trajectory.shape[0] == 0 and self._last_planner_control_mode != "yaw_delta")
        should_plan = self._last_trajectory.shape[0] == 0 or (
            observation.frame_id % max(int(getattr(self.args, "dual_request_gap_frames", 1)), 1) == 0
        )
        if should_plan:
            self.dual_planner.submit(
                DualPlannerInput(
                    frame_id=int(observation.frame_id),
                    rgb=observation.rgb,
                    depth=observation.depth,
                    sensor_meta=dict(observation.sensor_meta),
                    cam_pos=observation.cam_pos,
                    cam_quat=observation.cam_quat,
                    memory_context=observation.memory_context,
                    events={
                        "force_s2": force_s2,
                        "stuck": False,
                        "collision_risk": False,
                    },
                )
            )

        plan = self.dual_planner.consume_latest(last_seen_version=self._interactive_last_dual_plan_version)
        if plan is not None:
            self._interactive_last_dual_plan_version = int(plan.plan_version)
            self._last_dual_response_ts = time.perf_counter()
            planner_control = dict(plan.planner_control) if isinstance(plan.planner_control, dict) else {}
            if str(planner_control.get("mode", "trajectory")).strip().lower() == "stop" or bool(plan.stop):
                self._interactive_log(
                    "TASK",
                    "command_id={} completed goal_v={} traj_v={}".format(
                        int(self._interactive_active_command_id),
                        int(plan.goal_version),
                        int(plan.traj_version),
                    ),
                )
                self._activate_roaming(f"task complete command_id={int(self._interactive_active_command_id)}")
                return True
            self._accept_dual_plan(plan, interactive=True)

        success_calls, failed_calls, last_error, planner_latency_ms = self.dual_planner.snapshot_status()
        if int(failed_calls) > int(self._interactive_last_dual_failed_calls):
            self._stats = PlannerStats(
                successful_calls=int(success_calls),
                failed_calls=int(failed_calls),
                latency_ms=float(planner_latency_ms),
                last_error=str(last_error),
                last_plan_step=int(self._stats.last_plan_step),
            )
            self._interactive_last_dual_failed_calls = int(failed_calls)
            self._interactive_log(
                "TASK",
                f"command_id={int(self._interactive_active_command_id)} dual_step failed: {last_error}",
            )
            self._activate_roaming(f"task failure command_id={int(self._interactive_active_command_id)}")
            return False

        self._interactive_last_dual_failed_calls = int(failed_calls)
        self._stats = PlannerStats(
            successful_calls=int(success_calls),
            failed_calls=int(failed_calls),
            latency_ms=float(planner_latency_ms),
            last_error=str(last_error),
            last_plan_step=int(self._stats.last_plan_step),
        )
        return False

    def _consume_planner_latest(
        self,
        *,
        planner,
        frame_id: int,
        action_command: ActionCommand | None,
        goal_local_xy: np.ndarray | None = None,
        sensor_meta: dict[str, Any] | None = None,
    ) -> TrajectoryUpdate:
        deadline = time.time() + float(getattr(self.args, "plan_wait_timeout_sec", 0.5))
        latest: PlannerOutput | None = None
        while time.time() < deadline:
            latest = planner.consume_latest(self._last_plan_version)
            if latest is not None:
                break
            time.sleep(0.01)
        success, failed, error, latency_ms = planner.snapshot_status()
        if latest is not None:
            self._last_plan_version = int(latest.plan_version)
            self._last_trajectory = np.asarray(latest.trajectory_world, dtype=np.float32).copy()
            source_frame_id = int(latest.source_frame_id)
        else:
            source_frame_id = int(frame_id)
        self._reset_planner_control_state()
        self._stats = PlannerStats(
            successful_calls=int(success),
            failed_calls=int(failed),
            latency_ms=float(latency_ms),
            last_error=str(error),
            last_plan_step=source_frame_id,
        )
        return TrajectoryUpdate(
            trajectory_world=self._last_trajectory.copy(),
            plan_version=self._last_plan_version,
            stats=self._stats,
            source_frame_id=source_frame_id,
            goal_local_xy=goal_local_xy,
            action_command=action_command,
            stop=bool(action_command is None and self._last_trajectory.shape[0] == 0),
            sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
        )

    def _dual_reset(self, instruction: str, *, prefix: str) -> None:
        if self._dual_client is None:
            raise RuntimeError("dual client is not initialized")
        reset_rsp = self._dual_client.dual_reset(
            intrinsic=self._intrinsic.copy(),
            instruction=str(instruction),
            navdp_url=str(getattr(self.args, "server_url", "")),
            s1_period_sec=float(getattr(self.args, "s1_period_sec", 0.2)),
            s2_period_sec=float(getattr(self.args, "s2_period_sec", 1.0)),
            goal_ttl_sec=float(getattr(self.args, "goal_ttl_sec", 3.0)),
            traj_ttl_sec=float(getattr(self.args, "traj_ttl_sec", 1.5)),
            traj_max_stale_sec=float(getattr(self.args, "traj_max_stale_sec", 4.0)),
        )
        print(
            f"{prefix} dual_reset "
            f"algo={reset_rsp.algo} dual_server={getattr(self.args, 'dual_server_url', '')} "
            f"navdp_server={getattr(self.args, 'server_url', '')}"
        )

    def _accept_dual_plan(self, plan, *, interactive: bool = False) -> None:
        planner_control = dict(plan.planner_control) if isinstance(plan.planner_control, dict) else {}
        planner_mode = str(planner_control.get("mode", "trajectory")).strip().lower() or "trajectory"
        planner_yaw_delta = planner_control.get("yaw_delta_rad")
        if planner_yaw_delta is not None:
            planner_yaw_delta = float(planner_yaw_delta)
        planner_reason = str(planner_control.get("reason", ""))
        if planner_mode == "trajectory":
            if interactive:
                self._emit_interactive_trajectory(plan.trajectory_world)
            else:
                self._last_plan_version = int(plan.plan_version)
                self._last_trajectory = np.asarray(plan.trajectory_world, dtype=np.float32).copy()
        else:
            if interactive:
                self._emit_interactive_trajectory(np.zeros((0, 3), dtype=np.float32))
            else:
                self._last_plan_version = int(plan.plan_version)
                self._last_trajectory = np.zeros((0, 3), dtype=np.float32)
        self._last_planner_control_mode = planner_mode
        self._last_planner_yaw_delta_rad = planner_yaw_delta
        self._last_planner_control_reason = planner_reason
        self._last_used_cached_traj = bool(plan.used_cached_traj if planner_mode == "trajectory" else False)
        self._last_server_stale_sec = float(plan.stale_sec)
        self._last_goal_version = int(plan.goal_version)
        self._last_traj_version = int(plan.traj_version)
        self._last_system2_pixel_goal = None
        if plan.pixel_goal is not None:
            pixel_goal = np.asarray(plan.pixel_goal, dtype=np.float32).reshape(-1)
            if pixel_goal.shape[0] >= 2 and np.all(np.isfinite(pixel_goal[:2])):
                self._last_system2_pixel_goal = [int(round(float(pixel_goal[0]))), int(round(float(pixel_goal[1])))]
        self._stats = PlannerStats(
            successful_calls=int(plan.successful_calls),
            failed_calls=int(plan.failed_calls),
            latency_ms=float(plan.latency_ms),
            last_error=str(plan.last_error),
            last_plan_step=int(plan.source_frame_id),
        )

    def _emit_interactive_trajectory(self, trajectory_world: np.ndarray) -> None:
        traj = np.asarray(trajectory_world, dtype=np.float32)
        if traj.size == 0:
            traj = np.zeros((0, 3), dtype=np.float32)
        self._interactive_session_plan_version += 1
        self._last_plan_version = int(self._interactive_session_plan_version)
        self._last_trajectory = traj.copy()

    def _interactive_clear_active_task(self) -> None:
        self._interactive_active_command_id = -1
        self._interactive_active_instruction = ""

    def _interactive_log(self, phase: str, message: str) -> None:
        print(f"[G1_INTERACTIVE][{phase}] {message}")

    def _reset_planner_control_state(self) -> None:
        self._last_planner_control_mode = None
        self._last_planner_yaw_delta_rad = None
        self._last_planner_control_reason = ""
        self._last_used_cached_traj = False

    def _build_update(
        self,
        *,
        frame_id: int,
        action_command: ActionCommand | None,
        stop: bool = False,
        sensor_meta: dict[str, Any] | None = None,
    ) -> TrajectoryUpdate:
        interactive_phase = None
        interactive_command_id = -1
        interactive_instruction = ""
        if self.mode == "interactive":
            interactive_phase = str(self._interactive_state)
            interactive_command_id = int(self._interactive_active_command_id)
            interactive_instruction = str(self._interactive_active_instruction)
        return TrajectoryUpdate(
            trajectory_world=np.asarray(self._last_trajectory, dtype=np.float32).copy(),
            plan_version=int(self._last_plan_version),
            stats=self._stats,
            source_frame_id=int(frame_id),
            goal_local_xy=self._last_goal_local_xy.copy() if self.mode == "pointgoal" else None,
            action_command=action_command,
            stop=bool(stop),
            planner_control_mode=self._last_planner_control_mode,
            planner_yaw_delta_rad=self._last_planner_yaw_delta_rad,
            stale_sec=float(self._last_server_stale_sec),
            goal_version=int(self._last_goal_version),
            traj_version=int(self._last_traj_version),
            used_cached_traj=bool(self._last_used_cached_traj),
            sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
            interactive_phase=interactive_phase,
            interactive_command_id=interactive_command_id,
            interactive_instruction=interactive_instruction,
        )

    def viewer_overlay_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "trajectory_world": np.asarray(self._last_trajectory, dtype=np.float32).tolist(),
            "plan_version": int(self._last_plan_version),
        }
        if self.mode in {"dual", "interactive"}:
            state["goal_version"] = int(self._last_goal_version)
            state["traj_version"] = int(self._last_traj_version)
            state["stale_sec"] = float(self._last_server_stale_sec)
            state["planner_control_mode"] = self._last_planner_control_mode
            state["planner_yaw_delta_rad"] = self._last_planner_yaw_delta_rad
            state["planner_control_reason"] = str(self._last_planner_control_reason)
            if self._last_system2_pixel_goal is not None:
                state["system2_pixel_goal"] = list(self._last_system2_pixel_goal)
        if self.mode == "interactive":
            state["interactive_phase"] = str(self._interactive_state)
            state["interactive_command_id"] = int(self._interactive_active_command_id)
            state["interactive_instruction"] = str(self._interactive_active_instruction)
        return state

    def _default_navdp_client_factory(self, intrinsic: np.ndarray, args: argparse.Namespace) -> InProcessNavDPClient:
        return create_inprocess_navdp_client(
            intrinsic=intrinsic,
            backend=str(getattr(args, "navdp_backend", "auto")),
            checkpoint_path=str(getattr(args, "navdp_checkpoint", "")),
            device=str(getattr(args, "navdp_device", "cpu")),
            amp=bool(getattr(args, "navdp_amp", False)),
            amp_dtype=str(getattr(args, "navdp_amp_dtype", "float16")),
            tf32=bool(getattr(args, "navdp_tf32", False)),
            stop_threshold=float(getattr(args, "stop_threshold", -3.0)),
        )

    def _default_dual_client_factory(self, args: argparse.Namespace) -> DualSystemClient:
        return DualSystemClient(
            DualSystemClientConfig(
                base_url=str(getattr(args, "dual_server_url", "http://127.0.0.1:8890")),
                timeout_sec=float(getattr(args, "timeout_sec", 5.0)),
                reset_timeout_sec=float(getattr(args, "reset_timeout_sec", 15.0)),
                retry=int(getattr(args, "retry", 1)),
            )
        )

    @staticmethod
    def _default_intrinsic(width: int, height: int) -> np.ndarray:
        w = max(int(width), 1)
        h = max(int(height), 1)
        focal = float(max(w, h))
        return np.asarray(
            [[focal, 0.0, w * 0.5], [0.0, focal, h * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
