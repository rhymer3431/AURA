from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from adapters.sensors.d455_sensor import D455SensorAdapter, D455SensorAdapterConfig
from common.geometry import world_goal_to_robot_frame
from control.async_planners import AsyncNoGoalPlanner, AsyncPointGoalPlanner, NoGoalPlannerInput, PlannerInput, PlannerOutput
from inference.navdp import InProcessNavDPClient, create_inprocess_navdp_client
from ipc.messages import ActionCommand


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


@dataclass(frozen=True)
class TrajectoryUpdate:
    trajectory_world: np.ndarray
    plan_version: int
    stats: PlannerStats
    source_frame_id: int
    goal_local_xy: np.ndarray | None = None
    action_command: ActionCommand | None = None
    stop: bool = False


class PlanningSession:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        sensor_factory: Callable[[D455SensorAdapterConfig], D455SensorAdapter] | None = None,
        navdp_client_factory: Callable[[np.ndarray, argparse.Namespace], Any] | None = None,
    ) -> None:
        self.args = args
        self.sensor_factory = sensor_factory or (lambda cfg: D455SensorAdapter(cfg))
        self.navdp_client_factory = navdp_client_factory or self._default_navdp_client_factory
        self.sensor: D455SensorAdapter | None = None
        self.navdp_client: Any | None = None
        self.pointgoal_planner: AsyncPointGoalPlanner | None = None
        self.nogoal_planner: AsyncNoGoalPlanner | None = None
        self._last_plan_version = -1
        self._last_trajectory = np.zeros((0, 3), dtype=np.float32)
        self.last_sensor_init_report: dict[str, Any] = {}

    @property
    def navdp_backend_name(self) -> str:
        if self.navdp_client is None:
            return ""
        return str(getattr(self.navdp_client, "backend_name", getattr(self.navdp_client, "backend_impl", "")))

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
    ) -> None:
        self.navdp_client = navdp_client or self.navdp_client_factory(np.asarray(intrinsic, dtype=np.float32), self.args)
        self.navdp_client.navigator_reset(np.asarray(intrinsic, dtype=np.float32), batch_size=1)
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

    def shutdown(self) -> None:
        if self.pointgoal_planner is not None:
            self.pointgoal_planner.stop()
        if self.nogoal_planner is not None:
            self.nogoal_planner.stop()

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
            intrinsic=self.sensor.intrinsic,
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
            )
        if action_command.action_type == "LOCAL_SEARCH":
            return self._run_nogoal(observation, action_command=action_command)
        if action_command.target_pose_xyz is None:
            return TrajectoryUpdate(
                trajectory_world=self._last_trajectory.copy(),
                plan_version=self._last_plan_version,
                stats=PlannerStats(failed_calls=1, last_error="target_pose_xyz is required", last_plan_step=int(observation.frame_id)),
                source_frame_id=int(observation.frame_id),
                action_command=action_command,
                stop=True,
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
        return self._consume_latest(
            planner=self.pointgoal_planner,
            frame_id=observation.frame_id,
            action_command=action_command,
            goal_local_xy=np.asarray(goal_local_xy, dtype=np.float32),
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
        return self._consume_latest(
            planner=self.nogoal_planner,
            frame_id=observation.frame_id,
            action_command=action_command,
        )

    def _consume_latest(
        self,
        *,
        planner,
        frame_id: int,
        action_command: ActionCommand | None,
        goal_local_xy: np.ndarray | None = None,
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
        return TrajectoryUpdate(
            trajectory_world=self._last_trajectory.copy(),
            plan_version=self._last_plan_version,
            stats=PlannerStats(
                successful_calls=int(success),
                failed_calls=int(failed),
                latency_ms=float(latency_ms),
                last_error=str(error),
                last_plan_step=source_frame_id,
            ),
            source_frame_id=source_frame_id,
            goal_local_xy=goal_local_xy,
            action_command=action_command,
            stop=bool(action_command is None and self._last_trajectory.shape[0] == 0),
        )

    def _default_navdp_client_factory(self, intrinsic: np.ndarray, args: argparse.Namespace) -> InProcessNavDPClient:
        client = create_inprocess_navdp_client(
            intrinsic=intrinsic,
            backend=str(getattr(args, "navdp_backend", "auto")),
            checkpoint_path=str(getattr(args, "navdp_checkpoint", "")),
            device=str(getattr(args, "navdp_device", "cpu")),
            amp=bool(getattr(args, "navdp_amp", False)),
            amp_dtype=str(getattr(args, "navdp_amp_dtype", "float16")),
            tf32=bool(getattr(args, "navdp_tf32", False)),
            stop_threshold=float(getattr(args, "stop_threshold", -3.0)),
        )
        return client

    @staticmethod
    def _default_intrinsic(width: int, height: int) -> np.ndarray:
        w = max(int(width), 1)
        h = max(int(height), 1)
        focal = float(max(w, h))
        return np.asarray(
            [[focal, 0.0, w * 0.5], [0.0, focal, h * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
