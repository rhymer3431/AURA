from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from common.geometry import quat_wxyz_to_yaw, world_goal_to_robot_frame, wrap_to_pi


@dataclass(frozen=True)
class TrajectoryTrackerConfig:
    max_vx: float = 0.5
    max_vy: float = 0.3
    max_wz: float = 0.8
    lookahead_distance_m: float = 0.6
    heading_slowdown_rad: float = 0.6
    traj_stale_timeout_sec: float = 1.5
    cmd_accel_limit: float = 1.0
    cmd_yaw_accel_limit: float = 1.5
    handoff_reset_distance_m: float = 0.35
    handoff_reset_heading_rad: float = 0.5


@dataclass(frozen=True)
class TrackerResult:
    command: np.ndarray
    progress_idx: int
    target_idx: int
    heading_error_rad: float
    stale: bool


@dataclass(frozen=True)
class TrackerPoseTarget:
    pose_command_b: np.ndarray
    progress_idx: int
    target_idx: int
    heading_error_rad: float
    stale: bool
    target_xy_world: np.ndarray
    target_yaw_world: float


@dataclass(frozen=True)
class TrackerHandoff:
    reset_progress: bool
    seed_progress_idx: int | None
    projected_progress_idx: int
    target_shift_m: float
    heading_delta_rad: float


class TrajectoryTracker:
    def __init__(self, config: TrajectoryTrackerConfig):
        self.config = config
        self._trajectory_world = np.zeros((0, 3), dtype=np.float32)
        self._plan_version = -1
        self._progress_idx = 0
        self._last_trajectory_time = 0.0
        self._last_command = np.zeros(3, dtype=np.float32)
        self._last_command_time = 0.0

    @property
    def plan_version(self) -> int:
        return int(self._plan_version)

    @property
    def progress_idx(self) -> int:
        return int(self._progress_idx)

    def set_trajectory(
        self,
        trajectory_world: np.ndarray,
        *,
        plan_version: int,
        timestamp: float | None = None,
        reset_progress: bool = True,
        seed_progress_idx: int | None = None,
    ) -> None:
        traj = self._normalize_trajectory(trajectory_world)
        if traj.size == 0:
            self.clear(timestamp=timestamp)
            self._plan_version = int(plan_version)
            return
        self._trajectory_world = traj.copy()
        self._plan_version = int(plan_version)
        if reset_progress:
            self._progress_idx = 0
        else:
            base_idx = self._progress_idx if seed_progress_idx is None else int(seed_progress_idx)
            self._progress_idx = int(np.clip(base_idx, 0, self._trajectory_world.shape[0] - 1))
        self._last_trajectory_time = time.monotonic() if timestamp is None else float(timestamp)

    def compute_handoff(
        self,
        trajectory_world: np.ndarray,
        *,
        position_w: np.ndarray,
    ) -> TrackerHandoff:
        new_traj = self._normalize_trajectory(trajectory_world)
        if self._trajectory_world.shape[0] == 0 or new_traj.shape[0] == 0:
            return TrackerHandoff(
                reset_progress=True,
                seed_progress_idx=None,
                projected_progress_idx=0,
                target_shift_m=float("inf"),
                heading_delta_rad=float(np.pi),
            )

        robot_pos = np.asarray(position_w, dtype=np.float32).reshape(-1)
        robot_xy = robot_pos[:2]
        old_progress_idx = self._advanced_progress_idx(self._trajectory_world, self._progress_idx, robot_xy)
        old_target_idx = self._select_target_idx_for(self._trajectory_world, old_progress_idx)
        if old_target_idx < 0:
            return TrackerHandoff(
                reset_progress=True,
                seed_progress_idx=None,
                projected_progress_idx=0,
                target_shift_m=float("inf"),
                heading_delta_rad=float(np.pi),
            )
        old_target_xy = self._trajectory_world[old_target_idx, :2].astype(np.float32).copy()
        old_target_yaw = self._select_target_yaw_for(self._trajectory_world, old_target_idx, robot_xy)

        min_progress_idx = min(old_progress_idx, new_traj.shape[0] - 1)
        projected_progress_idx = self._project_progress_idx(new_traj, robot_xy, min_idx=min_progress_idx)
        new_target_idx = self._select_target_idx_for(new_traj, projected_progress_idx)
        if new_target_idx < 0:
            return TrackerHandoff(
                reset_progress=True,
                seed_progress_idx=None,
                projected_progress_idx=int(projected_progress_idx),
                target_shift_m=float("inf"),
                heading_delta_rad=float(np.pi),
            )
        new_target_xy = new_traj[new_target_idx, :2].astype(np.float32).copy()
        new_target_yaw = self._select_target_yaw_for(new_traj, new_target_idx, robot_xy)
        target_shift_m = float(np.linalg.norm(new_target_xy - old_target_xy))
        heading_delta_rad = abs(wrap_to_pi(float(new_target_yaw) - float(old_target_yaw)))
        reset_progress = bool(
            target_shift_m > float(self.config.handoff_reset_distance_m)
            or heading_delta_rad > float(self.config.handoff_reset_heading_rad)
        )
        return TrackerHandoff(
            reset_progress=reset_progress,
            seed_progress_idx=None if reset_progress else int(projected_progress_idx),
            projected_progress_idx=int(projected_progress_idx),
            target_shift_m=target_shift_m,
            heading_delta_rad=heading_delta_rad,
        )

    def clear(self, *, timestamp: float | None = None) -> None:
        self._trajectory_world = np.zeros((0, 3), dtype=np.float32)
        self._progress_idx = 0
        self._last_trajectory_time = time.monotonic() if timestamp is None else float(timestamp)
        self._last_command = np.zeros(3, dtype=np.float32)
        self._last_command_time = self._last_trajectory_time

    def is_stale(self, *, now: float | None = None) -> bool:
        if self._trajectory_world.shape[0] == 0:
            return True
        current_time = time.monotonic() if now is None else float(now)
        return (current_time - self._last_trajectory_time) > float(self.config.traj_stale_timeout_sec)

    def compute_command(
        self,
        position_w: np.ndarray,
        quat_wxyz: np.ndarray,
        *,
        now: float | None = None,
        force_stop: bool = False,
    ) -> TrackerResult:
        current_time = time.monotonic() if now is None else float(now)
        if force_stop or self._trajectory_world.shape[0] == 0 or self.is_stale(now=current_time):
            self._last_command = np.zeros(3, dtype=np.float32)
            self._last_command_time = current_time
            return TrackerResult(
                command=self._last_command.copy(),
                progress_idx=int(self._progress_idx),
                target_idx=-1,
                heading_error_rad=0.0,
                stale=True,
            )

        robot_pos = np.asarray(position_w, dtype=np.float32).reshape(-1)
        robot_xy = robot_pos[:2]
        robot_yaw = quat_wxyz_to_yaw(np.asarray(quat_wxyz, dtype=np.float32).reshape(-1))

        self._advance_progress(robot_xy)
        target_idx = self._select_target_idx()
        target_xy = self._trajectory_world[target_idx, :2]
        err_b = world_goal_to_robot_frame(goal_xy=target_xy, robot_xy=robot_xy, robot_yaw=float(robot_yaw))
        heading_error = wrap_to_pi(float(np.arctan2(float(err_b[1]), float(err_b[0]))))

        raw_command = np.asarray(
            [
                np.clip(float(err_b[0]), -float(self.config.max_vx), float(self.config.max_vx)),
                np.clip(float(err_b[1]), -float(self.config.max_vy), float(self.config.max_vy)),
                np.clip(1.5 * heading_error, -float(self.config.max_wz), float(self.config.max_wz)),
            ],
            dtype=np.float32,
        )

        if abs(heading_error) > float(self.config.heading_slowdown_rad):
            excess = abs(heading_error) - float(self.config.heading_slowdown_rad)
            span = max(np.pi - float(self.config.heading_slowdown_rad), 1.0e-6)
            slowdown = max(0.0, 1.0 - (excess / span))
            raw_command[:2] *= float(slowdown)

        if target_idx == self._trajectory_world.shape[0] - 1 and np.linalg.norm(err_b[:2]) < 0.05:
            raw_command[:2] = 0.0
            raw_command[2] = 0.0

        command = self._apply_slew_limit(raw_command, now=current_time)
        self._last_command = command.copy()
        self._last_command_time = current_time
        return TrackerResult(
            command=command,
            progress_idx=int(self._progress_idx),
            target_idx=int(target_idx),
            heading_error_rad=float(heading_error),
            stale=False,
        )

    def compute_target_pose(
        self,
        position_w: np.ndarray,
        quat_wxyz: np.ndarray,
        *,
        now: float | None = None,
        force_stop: bool = False,
    ) -> TrackerPoseTarget:
        current_time = time.monotonic() if now is None else float(now)
        zero_pose = np.zeros(4, dtype=np.float32)
        zero_xy = np.zeros(2, dtype=np.float32)
        if force_stop or self._trajectory_world.shape[0] == 0 or self.is_stale(now=current_time):
            return TrackerPoseTarget(
                pose_command_b=zero_pose,
                progress_idx=int(self._progress_idx),
                target_idx=-1,
                heading_error_rad=0.0,
                stale=True,
                target_xy_world=zero_xy,
                target_yaw_world=0.0,
            )

        robot_pos = np.asarray(position_w, dtype=np.float32).reshape(-1)
        robot_xy = robot_pos[:2]
        robot_yaw = quat_wxyz_to_yaw(np.asarray(quat_wxyz, dtype=np.float32).reshape(-1))

        self._advance_progress(robot_xy)
        target_idx = self._select_target_idx()
        target_xy = self._trajectory_world[target_idx, :2].astype(np.float32).copy()
        target_yaw_world = self._select_target_yaw(target_idx, robot_xy)
        target_xy_b = world_goal_to_robot_frame(goal_xy=target_xy, robot_xy=robot_xy, robot_yaw=float(robot_yaw))
        heading_error = wrap_to_pi(float(target_yaw_world) - float(robot_yaw))
        pose_command_b = np.asarray(
            [float(target_xy_b[0]), float(target_xy_b[1]), 0.0, float(heading_error)],
            dtype=np.float32,
        )
        return TrackerPoseTarget(
            pose_command_b=pose_command_b,
            progress_idx=int(self._progress_idx),
            target_idx=int(target_idx),
            heading_error_rad=float(heading_error),
            stale=False,
            target_xy_world=target_xy,
            target_yaw_world=float(target_yaw_world),
        )

    def _advance_progress(self, robot_xy: np.ndarray) -> None:
        self._progress_idx = self._advanced_progress_idx(self._trajectory_world, self._progress_idx, robot_xy)

    def _select_target_idx(self) -> int:
        return self._select_target_idx_for(self._trajectory_world, self._progress_idx)

    def _select_target_yaw(self, target_idx: int, robot_xy: np.ndarray) -> float:
        return self._select_target_yaw_for(self._trajectory_world, target_idx, robot_xy)

    @staticmethod
    def _normalize_trajectory(trajectory_world: np.ndarray) -> np.ndarray:
        traj = np.asarray(trajectory_world, dtype=np.float32)
        if traj.ndim != 2 or (traj.shape[0] > 0 and traj.shape[1] < 2):
            raise ValueError(f"trajectory_world must be [N,2+], got shape={traj.shape}")
        if traj.shape[1] == 2:
            traj = np.concatenate([traj, np.zeros((traj.shape[0], 1), dtype=np.float32)], axis=1)
        return traj

    @staticmethod
    def _advanced_progress_idx(trajectory_world: np.ndarray, current_progress_idx: int, robot_xy: np.ndarray) -> int:
        if trajectory_world.shape[0] == 0:
            return 0
        start_idx = int(np.clip(int(current_progress_idx), 0, trajectory_world.shape[0] - 1))
        remaining = trajectory_world[start_idx:, :2]
        if remaining.shape[0] == 0:
            return int(trajectory_world.shape[0] - 1)
        distances = np.linalg.norm(remaining - robot_xy.reshape(1, 2), axis=1)
        return int(start_idx + int(np.argmin(distances)))

    def _select_target_idx_for(self, trajectory_world: np.ndarray, progress_idx: int) -> int:
        if trajectory_world.shape[0] == 0:
            return -1
        if progress_idx >= trajectory_world.shape[0] - 1:
            return int(trajectory_world.shape[0] - 1)
        segments = np.diff(trajectory_world[progress_idx:, :2], axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        rel_target_idx = int(np.searchsorted(cumulative, float(self.config.lookahead_distance_m), side="left"))
        return min(int(progress_idx) + rel_target_idx, trajectory_world.shape[0] - 1)

    @staticmethod
    def _select_target_yaw_for(trajectory_world: np.ndarray, target_idx: int, robot_xy: np.ndarray) -> float:
        if target_idx < 0 or trajectory_world.shape[0] == 0:
            return 0.0
        if trajectory_world.shape[0] == 1:
            tangent_xy = trajectory_world[0, :2] - robot_xy[:2]
        elif target_idx < trajectory_world.shape[0] - 1:
            tangent_xy = trajectory_world[target_idx + 1, :2] - trajectory_world[target_idx, :2]
        else:
            tangent_xy = trajectory_world[target_idx, :2] - trajectory_world[target_idx - 1, :2]
        if float(np.linalg.norm(tangent_xy[:2])) <= 1.0e-6:
            tangent_xy = trajectory_world[target_idx, :2] - robot_xy[:2]
        if float(np.linalg.norm(tangent_xy[:2])) <= 1.0e-6:
            return 0.0
        return float(np.arctan2(float(tangent_xy[1]), float(tangent_xy[0])))

    @staticmethod
    def _project_progress_idx(trajectory_world: np.ndarray, robot_xy: np.ndarray, *, min_idx: int) -> int:
        if trajectory_world.shape[0] == 0:
            return 0
        lower_bound = int(np.clip(int(min_idx), 0, trajectory_world.shape[0] - 1))
        if trajectory_world.shape[0] == 1 or lower_bound >= trajectory_world.shape[0] - 1:
            return lower_bound

        best_idx = lower_bound
        best_dist_sq = float("inf")
        for idx in range(lower_bound, trajectory_world.shape[0] - 1):
            start_xy = trajectory_world[idx, :2]
            end_xy = trajectory_world[idx + 1, :2]
            segment = end_xy - start_xy
            seg_len_sq = float(np.dot(segment, segment))
            if seg_len_sq <= 1.0e-9:
                projection = start_xy
                t = 0.0
            else:
                t = float(np.clip(np.dot(robot_xy - start_xy, segment) / seg_len_sq, 0.0, 1.0))
                projection = start_xy + (t * segment)
            dist_sq = float(np.sum((robot_xy - projection) ** 2))
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = idx + 1 if t >= (1.0 - 1.0e-4) else idx

        final_distances = np.linalg.norm(trajectory_world[lower_bound:, :2] - robot_xy.reshape(1, 2), axis=1)
        final_vertex_idx = int(lower_bound + int(np.argmin(final_distances)))
        final_vertex_dist_sq = float(np.sum((trajectory_world[final_vertex_idx, :2] - robot_xy) ** 2))
        if final_vertex_dist_sq < best_dist_sq:
            return final_vertex_idx
        return int(best_idx)

    def _apply_slew_limit(self, command: np.ndarray, *, now: float) -> np.ndarray:
        if self._last_command_time <= 0.0:
            return command.copy()
        dt = max(float(now) - float(self._last_command_time), 0.0)
        if dt <= 0.0:
            return command.copy()

        limited = command.copy()
        linear_delta = float(self.config.cmd_accel_limit) * dt
        yaw_delta = float(self.config.cmd_yaw_accel_limit) * dt
        limited[0] = np.clip(limited[0], self._last_command[0] - linear_delta, self._last_command[0] + linear_delta)
        limited[1] = np.clip(limited[1], self._last_command[1] - linear_delta, self._last_command[1] + linear_delta)
        limited[2] = np.clip(limited[2], self._last_command[2] - yaw_delta, self._last_command[2] + yaw_delta)
        return limited
