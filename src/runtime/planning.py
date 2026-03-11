from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from adapters.d455_sensor import D455SensorAdapter, D455SensorAdapterConfig
from adapters.dual_http import DualSystemClient, DualSystemClientConfig
from adapters.navdp_http import NavDPClient, NavDPClientConfig
from common.geometry import world_goal_to_camera_pointgoal
from common.scene import place_demo_object, place_goal_marker
from control.async_planners import (
    AsyncDualPlanner,
    AsyncNoGoalPlanner,
    AsyncPointGoalPlanner,
    DualPlannerInput,
    DualPlannerOutput,
    NoGoalPlannerInput,
    PlannerInput,
    PlannerOutput,
)


@dataclass(frozen=True)
class PlannerStats:
    successful_calls: int = 0
    failed_calls: int = 0
    latency_ms: float = 0.0
    last_error: str = ""
    last_plan_step: int = -1


@dataclass(frozen=True)
class TrajectoryUpdate:
    trajectory_world: np.ndarray
    stop: bool
    plan_version: int
    stale_sec: float
    stats: PlannerStats
    source_frame_id: int
    goal_version: int = -1
    traj_version: int = -1
    used_cached_traj: bool = False
    goal_local_xy: np.ndarray | None = None
    sensor_meta: dict[str, Any] | None = None
    interactive_phase: str | None = None
    interactive_command_id: int = -1
    interactive_instruction: str = ""


@dataclass(frozen=True)
class DemoObjectState:
    prim_path: str
    world_xyz: np.ndarray
    size_m: float
    stop_radius_m: float


def _build_navdp_client(args: argparse.Namespace) -> NavDPClient:
    client_cfg = NavDPClientConfig(
        base_url=str(args.server_url),
        timeout_sec=float(args.timeout_sec),
        reset_timeout_sec=float(args.reset_timeout_sec),
        retry=int(args.retry),
        stop_threshold=float(args.stop_threshold),
    )
    return NavDPClient(client_cfg)


def _build_dual_client(args: argparse.Namespace) -> DualSystemClient:
    dual_cfg = DualSystemClientConfig(
        base_url=str(args.dual_server_url),
        timeout_sec=float(args.timeout_sec),
        reset_timeout_sec=float(args.reset_timeout_sec),
        retry=int(args.retry),
    )
    return DualSystemClient(dual_cfg)


def _dual_reset(
    *,
    args: argparse.Namespace,
    sensor: D455SensorAdapter,
    dual_client: DualSystemClient,
    instruction: str,
    prefix: str,
) -> None:
    reset_rsp = dual_client.dual_reset(
        intrinsic=sensor.intrinsic,
        instruction=str(instruction),
        navdp_url=str(args.server_url),
        s1_period_sec=float(args.s1_period_sec),
        s2_period_sec=float(args.s2_period_sec),
        goal_ttl_sec=float(args.goal_ttl_sec),
        traj_ttl_sec=float(args.traj_ttl_sec),
        traj_max_stale_sec=float(args.traj_max_stale_sec),
    )
    print(
        f"{prefix} dual_reset "
        f"algo={reset_rsp.algo} dual_server={args.dual_server_url} navdp_server={args.server_url}"
    )


def _build_pointgoal_planner(args: argparse.Namespace, sensor: D455SensorAdapter, stage):
    client = _build_navdp_client(args)
    algo = client.navigator_reset(sensor.intrinsic, batch_size=1)
    print(f"[G1_POINTGOAL] navigator_reset algo={algo}")
    planner = AsyncPointGoalPlanner(client=client, use_trajectory_z=bool(args.use_trajectory_z))
    planner.start()

    goal_world_xy = np.asarray([float(args.goal_x), float(args.goal_y)], dtype=np.float32)
    marker_ok, marker_msg = place_goal_marker(stage, goal_world_xy)
    if marker_ok:
        print(
            f"[G1_POINTGOAL] goal marker placed: {marker_msg} "
            f"at ({goal_world_xy[0]:.3f}, {goal_world_xy[1]:.3f})"
        )
    else:
        print(f"[G1_POINTGOAL] goal marker skipped: {marker_msg}")
    return planner, goal_world_xy


def _spawn_demo_object(stage, args: argparse.Namespace) -> DemoObjectState | None:
    if not bool(args.spawn_demo_object):
        return None

    object_world_xy = np.asarray([float(args.demo_object_x), float(args.demo_object_y)], dtype=np.float32)
    ok, object_prim_path, object_world_xyz = place_demo_object(
        stage,
        object_world_xy,
        object_size_m=float(args.demo_object_size_m),
    )
    if not ok:
        raise RuntimeError(f"demo object creation failed: {object_prim_path}")

    demo_object = DemoObjectState(
        prim_path=str(object_prim_path),
        world_xyz=np.asarray(object_world_xyz, dtype=np.float32).copy(),
        size_m=float(args.demo_object_size_m),
        stop_radius_m=float(args.object_stop_radius_m),
    )
    print(
        "[G1_OBJECT_SEARCH] demo object placed "
        f"prim={demo_object.prim_path} "
        f"xyz=({float(demo_object.world_xyz[0]):.3f},"
        f"{float(demo_object.world_xyz[1]):.3f},"
        f"{float(demo_object.world_xyz[2]):.3f}) "
        f"size={demo_object.size_m:.3f}m stop_radius={demo_object.stop_radius_m:.3f}m"
    )
    return demo_object


def _build_dual_planner(args: argparse.Namespace, sensor: D455SensorAdapter):
    dual_client = _build_dual_client(args)
    _dual_reset(
        args=args,
        sensor=sensor,
        dual_client=dual_client,
        instruction=str(args.instruction),
        prefix="[G1_DUAL]",
    )
    planner = AsyncDualPlanner(client=dual_client)
    planner.start()
    return dual_client, planner


class PlannerSession:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sensor: D455SensorAdapter | None = None
        self.pointgoal_planner: AsyncPointGoalPlanner | None = None
        self.nogoal_planner: AsyncNoGoalPlanner | None = None
        self.dual_planner: AsyncDualPlanner | None = None
        self._nogoal_client: NavDPClient | None = None
        self._dual_client: DualSystemClient | None = None
        self.goal_world_xy: np.ndarray | None = None
        self.demo_object: DemoObjectState | None = None
        self.mode = str(args.planner_mode).lower()

        self._last_trajectory_world = np.zeros((0, 3), dtype=np.float32)
        self._last_plan_version = -1
        self._last_goal_local_xy = np.zeros(2, dtype=np.float32)
        self._last_server_stale_sec = -1.0
        self._last_goal_version = -1
        self._last_traj_version = -1
        self._last_dual_response_ts = time.perf_counter()
        self._stats = PlannerStats()

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
    def stats(self) -> PlannerStats:
        return self._stats

    def initialize(self, simulation_app, stage) -> None:
        sensor_cfg = D455SensorAdapterConfig(
            use_d455=True,
            image_width=int(self.args.image_width),
            image_height=int(self.args.image_height),
            depth_max_m=float(self.args.depth_max_m),
            strict_d455=bool(self.args.strict_d455),
            force_runtime_mount=bool(self.args.force_runtime_camera),
        )
        self.sensor = D455SensorAdapter(sensor_cfg)
        init_ok, init_msg = self.sensor.initialize(simulation_app, stage)
        print(f"[G1_POINTGOAL] D455 init: ok={init_ok} msg={init_msg}")
        if not init_ok:
            raise RuntimeError(f"D455 initialization failed: {init_msg}")

        self.demo_object = _spawn_demo_object(stage, self.args)
        if self.mode == "pointgoal":
            self.pointgoal_planner, self.goal_world_xy = _build_pointgoal_planner(self.args, self.sensor, stage)
            return

        if self.mode == "dual":
            self._dual_client, self.dual_planner = _build_dual_planner(self.args, self.sensor)
            return

        self._nogoal_client = _build_navdp_client(self.args)
        self.nogoal_planner = AsyncNoGoalPlanner(client=self._nogoal_client, use_trajectory_z=bool(self.args.use_trajectory_z))
        self.nogoal_planner.start()
        self._dual_client = _build_dual_client(self.args)
        self.dual_planner = AsyncDualPlanner(client=self._dual_client)
        self.dual_planner.start()
        if not self._activate_roaming("startup"):
            raise RuntimeError(self._stats.last_error or "interactive roaming initialization failed")

    def shutdown(self) -> None:
        for planner in (self.pointgoal_planner, self.nogoal_planner, self.dual_planner):
            if planner is not None:
                planner.stop()

    def no_response_sec(self, *, now: float | None = None) -> float:
        current = time.perf_counter() if now is None else float(now)
        if self.mode == "dual":
            return current - float(self._last_dual_response_ts)
        if self.mode == "interactive" and self._interactive_state == "task_active":
            return current - float(self._last_dual_response_ts)
        return 0.0

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

    def update(self, frame_id: int) -> TrajectoryUpdate:
        if self.sensor is None:
            raise RuntimeError("PlannerSession.initialize() must be called before update().")

        rgb, depth, sensor_meta = self.sensor.capture_rgbd_with_meta(None)
        if rgb is None or depth is None:
            self._stats = PlannerStats(
                successful_calls=int(self._stats.successful_calls),
                failed_calls=int(self._stats.failed_calls),
                latency_ms=float(self._stats.latency_ms),
                last_error=f"sensor frame unavailable meta={sensor_meta}",
                last_plan_step=int(self._stats.last_plan_step),
            )
            return self._build_update(frame_id=frame_id, sensor_meta=sensor_meta)

        cam_pos, cam_quat = self.sensor.get_rgb_camera_pose_world()
        if cam_pos is None or cam_quat is None:
            self._stats = PlannerStats(
                successful_calls=int(self._stats.successful_calls),
                failed_calls=int(self._stats.failed_calls),
                latency_ms=float(self._stats.latency_ms),
                last_error="missing camera world pose",
                last_plan_step=int(self._stats.last_plan_step),
            )
            return self._build_update(frame_id=frame_id, sensor_meta=sensor_meta)

        if self.mode == "pointgoal":
            return self._update_pointgoal(frame_id, rgb, depth, sensor_meta, cam_pos, cam_quat)
        if self.mode == "dual":
            return self._update_dual(frame_id, rgb, depth, sensor_meta, cam_pos, cam_quat)
        return self._update_interactive(frame_id, rgb, depth, sensor_meta, cam_pos, cam_quat)

    def _update_pointgoal(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        sensor_meta: dict[str, Any],
        cam_pos: np.ndarray,
        cam_quat: np.ndarray,
    ) -> TrajectoryUpdate:
        if self.pointgoal_planner is None or self.goal_world_xy is None:
            raise RuntimeError("pointgoal planner is not initialized")

        should_plan = self._last_trajectory_world.shape[0] == 0 or (
            frame_id % max(int(self.args.plan_interval_frames), 1) == 0
        )
        if should_plan:
            goal_world = np.asarray(
                [self.goal_world_xy[0], self.goal_world_xy[1], np.asarray(cam_pos, dtype=np.float32)[2]],
                dtype=np.float32,
            )
            self._last_goal_local_xy = world_goal_to_camera_pointgoal(goal_world, cam_pos, cam_quat)
            self.pointgoal_planner.submit(
                PlannerInput(
                    frame_id=int(frame_id),
                    local_goal_xy=np.asarray(self._last_goal_local_xy, dtype=np.float32),
                    rgb=np.asarray(rgb, dtype=np.uint8),
                    depth=np.asarray(depth, dtype=np.float32),
                    sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
                    cam_pos=np.asarray(cam_pos, dtype=np.float32),
                    cam_quat=np.asarray(cam_quat, dtype=np.float32),
                )
            )

        plan = self.pointgoal_planner.consume_latest(last_seen_version=self._last_plan_version)
        if plan is not None:
            self._accept_plan(plan)

        success_calls, failed_calls, last_error, planner_latency_ms = self.pointgoal_planner.snapshot_status()
        self._stats = PlannerStats(
            successful_calls=int(success_calls),
            failed_calls=int(failed_calls),
            latency_ms=float(planner_latency_ms),
            last_error=str(last_error),
            last_plan_step=int(self._stats.last_plan_step),
        )
        return self._build_update(frame_id=frame_id, sensor_meta=sensor_meta)

    def _update_dual(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        sensor_meta: dict[str, Any],
        cam_pos: np.ndarray,
        cam_quat: np.ndarray,
    ) -> TrajectoryUpdate:
        if self.dual_planner is None:
            raise RuntimeError("dual planner is not initialized")

        plan_stop = False
        should_plan = self._last_trajectory_world.shape[0] == 0 or (
            frame_id % max(int(self.args.dual_request_gap_frames), 1) == 0
        )
        if should_plan:
            self.dual_planner.submit(
                DualPlannerInput(
                    frame_id=int(frame_id),
                    rgb=np.asarray(rgb, dtype=np.uint8),
                    depth=np.asarray(depth, dtype=np.float32),
                    sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
                    cam_pos=np.asarray(cam_pos, dtype=np.float32),
                    cam_quat=np.asarray(cam_quat, dtype=np.float32),
                    memory_context=None,
                    events={
                        "force_s2": bool(self._last_trajectory_world.shape[0] == 0),
                        "stuck": False,
                        "collision_risk": False,
                    },
                )
            )

        plan = self.dual_planner.consume_latest(last_seen_version=self._last_plan_version)
        if plan is not None:
            self._last_dual_response_ts = time.perf_counter()
            self._accept_dual_plan(plan)
            plan_stop = bool(plan.stop)

        success_calls, failed_calls, last_error, planner_latency_ms = self.dual_planner.snapshot_status()
        self._stats = PlannerStats(
            successful_calls=int(success_calls),
            failed_calls=int(failed_calls),
            latency_ms=float(planner_latency_ms),
            last_error=str(last_error),
            last_plan_step=int(self._stats.last_plan_step),
        )
        return self._build_update(frame_id=frame_id, stop=plan_stop, sensor_meta=sensor_meta)

    def _update_interactive(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        sensor_meta: dict[str, Any],
        cam_pos: np.ndarray,
        cam_quat: np.ndarray,
    ) -> TrajectoryUpdate:
        self._consume_interactive_controls()

        if self._interactive_state == "roaming":
            self._update_interactive_roaming(frame_id, rgb, depth, sensor_meta, cam_pos, cam_quat)
        else:
            self._update_interactive_task(frame_id, rgb, depth, sensor_meta, cam_pos, cam_quat)

        return self._build_update(frame_id=frame_id, sensor_meta=sensor_meta)

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
        if self.sensor is None or self._nogoal_client is None or self.nogoal_planner is None:
            return False

        try:
            algo = self._nogoal_client.navigator_reset(self.sensor.intrinsic, batch_size=1)
        except Exception as exc:  # noqa: BLE001
            error = f"roaming navigator_reset failed: {type(exc).__name__}: {exc}"
            self._interactive_state = "roaming"
            self._last_server_stale_sec = -1.0
            self._last_goal_version = -1
            self._last_traj_version = -1
            self._stats = PlannerStats(
                successful_calls=0,
                failed_calls=1,
                latency_ms=0.0,
                last_error=error,
                last_plan_step=int(self._stats.last_plan_step),
            )
            self._interactive_log("ROAM", f"{reason} -> {error}")
            self._interactive_clear_active_task()
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
        self._interactive_state = "roaming"
        self._last_server_stale_sec = -1.0
        self._last_goal_version = -1
        self._last_traj_version = -1
        self._last_dual_response_ts = time.perf_counter()
        self._stats = PlannerStats()
        self._emit_interactive_trajectory(np.zeros((0, 3), dtype=np.float32))
        self._interactive_log("ROAM", f"state=roaming reason={reason} navdp_reset={algo}")
        return True

    def _activate_task(self, command_id: int, instruction: str) -> bool:
        if self.sensor is None or self._dual_client is None or self.dual_planner is None:
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

        try:
            _dual_reset(
                args=self.args,
                sensor=self.sensor,
                dual_client=self._dual_client,
                instruction=instruction,
                prefix="[G1_INTERACTIVE][TASK]",
            )
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
        self._last_dual_response_ts = time.perf_counter()
        self._stats = PlannerStats()
        self._interactive_log(
            "TASK",
            f"state=task_active command_id={int(command_id)} instruction={instruction!r}",
        )
        return True

    def _update_interactive_roaming(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        sensor_meta: dict[str, Any],
        cam_pos: np.ndarray,
        cam_quat: np.ndarray,
    ) -> None:
        if self.nogoal_planner is None:
            raise RuntimeError("interactive no-goal planner is not initialized")

        should_plan = self._last_trajectory_world.shape[0] == 0 or (
            frame_id % max(int(self.args.plan_interval_frames), 1) == 0
        )
        if should_plan:
            self.nogoal_planner.submit(
                NoGoalPlannerInput(
                    frame_id=int(frame_id),
                    rgb=np.asarray(rgb, dtype=np.uint8),
                    depth=np.asarray(depth, dtype=np.float32),
                    sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
                    cam_pos=np.asarray(cam_pos, dtype=np.float32),
                    cam_quat=np.asarray(cam_quat, dtype=np.float32),
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

    def _update_interactive_task(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        sensor_meta: dict[str, Any],
        cam_pos: np.ndarray,
        cam_quat: np.ndarray,
    ) -> None:
        if self.dual_planner is None:
            raise RuntimeError("interactive dual planner is not initialized")

        should_plan = self._last_trajectory_world.shape[0] == 0 or (
            frame_id % max(int(self.args.dual_request_gap_frames), 1) == 0
        )
        if should_plan:
            self.dual_planner.submit(
                DualPlannerInput(
                    frame_id=int(frame_id),
                    rgb=np.asarray(rgb, dtype=np.uint8),
                    depth=np.asarray(depth, dtype=np.float32),
                    sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
                    cam_pos=np.asarray(cam_pos, dtype=np.float32),
                    cam_quat=np.asarray(cam_quat, dtype=np.float32),
                    memory_context=None,
                    events={
                        "force_s2": bool(self._last_trajectory_world.shape[0] == 0),
                        "stuck": False,
                        "collision_risk": False,
                    },
                )
            )

        plan = self.dual_planner.consume_latest(last_seen_version=self._interactive_last_dual_plan_version)
        if plan is not None:
            self._interactive_last_dual_plan_version = int(plan.plan_version)
            self._last_dual_response_ts = time.perf_counter()
            if bool(plan.stop):
                self._interactive_log(
                    "TASK",
                    "command_id={} completed goal_v={} traj_v={}".format(
                        int(self._interactive_active_command_id),
                        int(plan.goal_version),
                        int(plan.traj_version),
                    ),
                )
                self._activate_roaming(f"task complete command_id={int(self._interactive_active_command_id)}")
                return
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
            return

        self._interactive_last_dual_failed_calls = int(failed_calls)
        self._stats = PlannerStats(
            successful_calls=int(success_calls),
            failed_calls=int(failed_calls),
            latency_ms=float(planner_latency_ms),
            last_error=str(last_error),
            last_plan_step=int(self._stats.last_plan_step),
        )

    def _accept_plan(self, plan: PlannerOutput) -> None:
        self._last_plan_version = int(plan.plan_version)
        self._last_trajectory_world = np.asarray(plan.trajectory_world, dtype=np.float32).copy()
        self._stats = PlannerStats(
            successful_calls=int(plan.successful_calls),
            failed_calls=int(plan.failed_calls),
            latency_ms=float(plan.latency_ms),
            last_error=str(plan.last_error),
            last_plan_step=int(plan.source_frame_id),
        )

    def _accept_dual_plan(self, plan: DualPlannerOutput, *, interactive: bool = False) -> None:
        if interactive:
            self._emit_interactive_trajectory(plan.trajectory_world)
        else:
            self._last_plan_version = int(plan.plan_version)
            self._last_trajectory_world = np.asarray(plan.trajectory_world, dtype=np.float32).copy()
        self._last_server_stale_sec = float(plan.stale_sec)
        self._last_goal_version = int(plan.goal_version)
        self._last_traj_version = int(plan.traj_version)
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
        self._last_trajectory_world = traj.copy()

    def _interactive_clear_active_task(self) -> None:
        self._interactive_active_command_id = -1
        self._interactive_active_instruction = ""

    def _interactive_log(self, phase: str, message: str) -> None:
        print(f"[G1_INTERACTIVE][{phase}] {message}")

    def _build_update(
        self,
        *,
        frame_id: int,
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
            trajectory_world=np.asarray(self._last_trajectory_world, dtype=np.float32).copy(),
            stop=bool(stop),
            plan_version=int(self._last_plan_version),
            stale_sec=float(self._last_server_stale_sec),
            stats=self._stats,
            source_frame_id=int(frame_id),
            goal_version=int(self._last_goal_version),
            traj_version=int(self._last_traj_version),
            used_cached_traj=bool(
                (self.mode == "dual" and self._last_trajectory_world.shape[0] > 0)
                or (self.mode == "interactive" and self._interactive_state == "task_active" and self._last_trajectory_world.shape[0] > 0)
            ),
            goal_local_xy=self._last_goal_local_xy.copy() if self.mode == "pointgoal" else None,
            sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
            interactive_phase=interactive_phase,
            interactive_command_id=interactive_command_id,
            interactive_instruction=interactive_instruction,
        )
