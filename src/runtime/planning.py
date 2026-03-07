from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from adapters.d455_sensor import D455SensorAdapter, D455SensorAdapterConfig
from adapters.dual_http import DualSystemClient, DualSystemClientConfig
from adapters.navdp_http import NavDPClient, NavDPClientConfig
from common.geometry import world_goal_to_camera_pointgoal
from common.scene import place_goal_marker
from control.async_planners import (
    AsyncDualPlanner,
    AsyncPointGoalPlanner,
    DualPlannerInput,
    DualPlannerOutput,
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


def _build_pointgoal_planner(args: argparse.Namespace, sensor: D455SensorAdapter, stage):
    client_cfg = NavDPClientConfig(
        base_url=str(args.server_url),
        timeout_sec=float(args.timeout_sec),
        reset_timeout_sec=float(args.reset_timeout_sec),
        retry=int(args.retry),
        stop_threshold=float(args.stop_threshold),
    )
    client = NavDPClient(client_cfg)
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


def _build_dual_planner(args: argparse.Namespace, sensor: D455SensorAdapter):
    dual_cfg = DualSystemClientConfig(
        base_url=str(args.dual_server_url),
        timeout_sec=float(args.timeout_sec),
        reset_timeout_sec=float(args.reset_timeout_sec),
        retry=int(args.retry),
    )
    dual_client = DualSystemClient(dual_cfg)
    reset_rsp = dual_client.dual_reset(
        intrinsic=sensor.intrinsic,
        instruction=str(args.instruction),
        navdp_url=str(args.server_url),
        s1_period_sec=float(args.s1_period_sec),
        s2_period_sec=float(args.s2_period_sec),
        goal_ttl_sec=float(args.goal_ttl_sec),
        traj_ttl_sec=float(args.traj_ttl_sec),
        traj_max_stale_sec=float(args.traj_max_stale_sec),
    )
    print(
        "[G1_DUAL] dual_reset "
        f"algo={reset_rsp.algo} dual_server={args.dual_server_url} navdp_server={args.server_url}"
    )
    planner = AsyncDualPlanner(client=dual_client)
    planner.start()
    return planner


class PlannerSession:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sensor: D455SensorAdapter | None = None
        self.planner: AsyncPointGoalPlanner | AsyncDualPlanner | None = None
        self.goal_world_xy: np.ndarray | None = None
        self.mode = str(args.planner_mode).lower()

        self._last_trajectory_world = np.zeros((0, 3), dtype=np.float32)
        self._last_plan_version = -1
        self._last_goal_local_xy = np.zeros(2, dtype=np.float32)
        self._last_server_stale_sec = -1.0
        self._last_goal_version = -1
        self._last_traj_version = -1
        self._last_dual_response_ts = time.perf_counter()
        self._stats = PlannerStats()

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

        if self.mode == "pointgoal":
            self.planner, self.goal_world_xy = _build_pointgoal_planner(self.args, self.sensor, stage)
        else:
            self.planner = _build_dual_planner(self.args, self.sensor)

    def shutdown(self) -> None:
        if self.planner is not None:
            self.planner.stop()

    def no_response_sec(self, *, now: float | None = None) -> float:
        if self.mode != "dual":
            return 0.0
        current = time.perf_counter() if now is None else float(now)
        return current - float(self._last_dual_response_ts)

    def update(self, frame_id: int) -> TrajectoryUpdate:
        if self.sensor is None or self.planner is None:
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

        plan_stop = False
        if self.mode == "pointgoal":
            assert isinstance(self.planner, AsyncPointGoalPlanner)
            assert self.goal_world_xy is not None
            should_plan = self._last_trajectory_world.shape[0] == 0 or (
                frame_id % max(int(self.args.plan_interval_frames), 1) == 0
            )
            if should_plan:
                goal_world = np.asarray(
                    [self.goal_world_xy[0], self.goal_world_xy[1], np.asarray(cam_pos, dtype=np.float32)[2]],
                    dtype=np.float32,
                )
                self._last_goal_local_xy = world_goal_to_camera_pointgoal(goal_world, cam_pos, cam_quat)
                self.planner.submit(
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

            plan = self.planner.consume_latest(last_seen_version=self._last_plan_version)
            if plan is not None:
                assert isinstance(plan, PlannerOutput)
                self._last_plan_version = int(plan.plan_version)
                self._last_trajectory_world = np.asarray(plan.trajectory_world, dtype=np.float32).copy()
                self._stats = PlannerStats(
                    successful_calls=int(plan.successful_calls),
                    failed_calls=int(plan.failed_calls),
                    latency_ms=float(plan.latency_ms),
                    last_error=str(plan.last_error),
                    last_plan_step=int(plan.source_frame_id),
                )
        else:
            assert isinstance(self.planner, AsyncDualPlanner)
            should_plan = self._last_trajectory_world.shape[0] == 0 or (
                frame_id % max(int(self.args.dual_request_gap_frames), 1) == 0
            )
            if should_plan:
                self.planner.submit(
                    DualPlannerInput(
                        frame_id=int(frame_id),
                        rgb=np.asarray(rgb, dtype=np.uint8),
                        depth=np.asarray(depth, dtype=np.float32),
                        sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
                        cam_pos=np.asarray(cam_pos, dtype=np.float32),
                        cam_quat=np.asarray(cam_quat, dtype=np.float32),
                        events={
                            "force_s2": bool(self._last_trajectory_world.shape[0] == 0),
                            "stuck": False,
                            "collision_risk": False,
                        },
                    )
                )

            plan = self.planner.consume_latest(last_seen_version=self._last_plan_version)
            if plan is not None:
                assert isinstance(plan, DualPlannerOutput)
                self._last_dual_response_ts = time.perf_counter()
                self._last_plan_version = int(plan.plan_version)
                self._last_trajectory_world = np.asarray(plan.trajectory_world, dtype=np.float32).copy()
                self._last_server_stale_sec = float(plan.stale_sec)
                self._last_goal_version = int(plan.goal_version)
                self._last_traj_version = int(plan.traj_version)
                plan_stop = bool(plan.stop)
                self._stats = PlannerStats(
                    successful_calls=int(plan.successful_calls),
                    failed_calls=int(plan.failed_calls),
                    latency_ms=float(plan.latency_ms),
                    last_error=str(plan.last_error),
                    last_plan_step=int(plan.source_frame_id),
                )

        success_calls, failed_calls, last_error, planner_latency_ms = self.planner.snapshot_status()
        self._stats = PlannerStats(
            successful_calls=int(success_calls),
            failed_calls=int(failed_calls),
            latency_ms=float(planner_latency_ms),
            last_error=str(last_error),
            last_plan_step=int(self._stats.last_plan_step),
        )
        return self._build_update(frame_id=frame_id, stop=plan_stop, sensor_meta=sensor_meta)

    def _build_update(
        self,
        *,
        frame_id: int,
        stop: bool = False,
        sensor_meta: dict[str, Any] | None = None,
    ) -> TrajectoryUpdate:
        return TrajectoryUpdate(
            trajectory_world=np.asarray(self._last_trajectory_world, dtype=np.float32).copy(),
            stop=bool(stop),
            plan_version=int(self._last_plan_version),
            stale_sec=float(self._last_server_stale_sec),
            stats=self._stats,
            source_frame_id=int(frame_id),
            goal_version=int(self._last_goal_version),
            traj_version=int(self._last_traj_version),
            used_cached_traj=bool(self.mode == "dual" and self._last_trajectory_world.shape[0] > 0),
            goal_local_xy=self._last_goal_local_xy.copy() if self.mode == "pointgoal" else None,
            sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
        )
