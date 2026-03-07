from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from adapters.legacy_http.navdp_http import NavDPClient, NavDPClientConfig
from adapters.sensors.d455_sensor import D455SensorAdapter, D455SensorAdapterConfig
from common.geometry import world_goal_to_robot_frame
from control.async_planners import AsyncNoGoalPlanner, AsyncPointGoalPlanner, NoGoalPlannerInput, PlannerInput, PlannerOutput
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
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sensor: D455SensorAdapter | None = None
        self.pointgoal_planner: AsyncPointGoalPlanner | None = None
        self.nogoal_planner: AsyncNoGoalPlanner | None = None
        self._last_plan_version = -1
        self._last_trajectory = np.zeros((0, 3), dtype=np.float32)

    def initialize(self, simulation_app, stage) -> None:
        self.sensor = D455SensorAdapter(
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
        print(f"[PLANNING_SESSION] sensor init: ok={init_ok} msg={init_msg}")
        if not init_ok:
            raise RuntimeError(init_msg)

        client = NavDPClient(
            NavDPClientConfig(
                base_url=str(getattr(self.args, "server_url", "http://127.0.0.1:8888")),
                timeout_sec=float(getattr(self.args, "timeout_sec", 5.0)),
                reset_timeout_sec=float(getattr(self.args, "reset_timeout_sec", 15.0)),
                retry=int(getattr(self.args, "retry", 1)),
                stop_threshold=float(getattr(self.args, "stop_threshold", -3.0)),
            )
        )
        client.navigator_reset(self.sensor.intrinsic, batch_size=1)
        self.pointgoal_planner = AsyncPointGoalPlanner(client=client, use_trajectory_z=bool(getattr(self.args, "use_trajectory_z", False)), pointgoal_frame="robot")
        self.nogoal_planner = AsyncNoGoalPlanner(client=client, use_trajectory_z=bool(getattr(self.args, "use_trajectory_z", False)))
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
        if action_command is None or action_command.action_type in {"STOP", "LOOK_AT"}:
            return TrajectoryUpdate(
                trajectory_world=np.zeros((0, 3), dtype=np.float32),
                plan_version=self._last_plan_version,
                stats=PlannerStats(last_plan_step=int(frame_id)),
                source_frame_id=int(frame_id),
                action_command=action_command,
                stop=True,
            )
        if action_command.action_type == "LOCAL_SEARCH":
            return self._run_nogoal(observation)
        if action_command.target_pose_xyz is None:
            return TrajectoryUpdate(
                trajectory_world=self._last_trajectory.copy(),
                plan_version=self._last_plan_version,
                stats=PlannerStats(failed_calls=1, last_error="target_pose_xyz is required", last_plan_step=int(frame_id)),
                source_frame_id=int(frame_id),
                action_command=action_command,
                stop=True,
            )
        goal_local_xy = world_goal_to_robot_frame(
            goal_xy=np.asarray(action_command.target_pose_xyz[:2], dtype=np.float32),
            robot_xy=np.asarray(robot_pos_world[:2], dtype=np.float32),
            robot_yaw=float(robot_yaw),
        )
        return self._run_pointgoal(observation, goal_local_xy, robot_pos_world, robot_quat_wxyz, robot_yaw, action_command)

    def _run_pointgoal(
        self,
        observation: ExecutionObservation,
        goal_local_xy: np.ndarray,
        robot_pos_world: np.ndarray,
        robot_quat_wxyz: np.ndarray,
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

    def _run_nogoal(self, observation: ExecutionObservation) -> TrajectoryUpdate:
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
        return self._consume_latest(planner=self.nogoal_planner, frame_id=observation.frame_id, action_command=None)

    def _consume_latest(
        self,
        *,
        planner,
        frame_id: int,
        action_command: ActionCommand | None,
        goal_local_xy: np.ndarray | None = None,
    ) -> TrajectoryUpdate:
        deadline = time.time() + 0.5
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
