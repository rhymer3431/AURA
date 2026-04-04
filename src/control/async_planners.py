from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from common.geometry import normalize_navdp_trajectory, trajectory_camera_to_world, trajectory_local_to_world


@dataclass
class PlannerInput:
    frame_id: int
    local_goal_xy: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    sensor_meta: dict[str, Any]
    cam_pos: np.ndarray
    cam_quat: np.ndarray
    robot_pos: np.ndarray | None = None
    robot_yaw: float | None = None


@dataclass
class PlannerOutput:
    plan_version: int
    source_frame_id: int
    trajectory_world: np.ndarray
    latency_ms: float
    successful_calls: int
    failed_calls: int
    last_error: str


@dataclass
class NoGoalPlannerInput:
    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    sensor_meta: dict[str, Any]
    cam_pos: np.ndarray
    cam_quat: np.ndarray


class AsyncNoGoalPlanner:
    def __init__(self, client: NavDPPlannerClient, use_trajectory_z: bool):
        self._client = client
        self._use_trajectory_z = bool(use_trajectory_z)
        self._lock = threading.Lock()
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: NoGoalPlannerInput | None = None
        self._latest: PlannerOutput | None = None
        self._plan_version = -1
        self._successful_calls = 0
        self._failed_calls = 0
        self._last_error = ""
        self._last_latency_ms = 0.0
        self._generation = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, name="navdp-nogoal-planner", daemon=True)
        self._thread.start()

    def stop(self, timeout_sec: float = 2.0) -> None:
        self._stop_event.set()
        self._request_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(float(timeout_sec), 0.1))

    def reset_state(self) -> None:
        with self._lock:
            self._generation += 1
            self._pending = None
            self._latest = None
            self._plan_version = -1
            self._successful_calls = 0
            self._failed_calls = 0
            self._last_error = ""
            self._last_latency_ms = 0.0

    def submit(self, planner_input: NoGoalPlannerInput) -> None:
        with self._lock:
            self._pending = planner_input
        self._request_event.set()

    def consume_latest(self, last_seen_version: int) -> PlannerOutput | None:
        with self._lock:
            if self._latest is None:
                return None
            if self._latest.plan_version <= int(last_seen_version):
                return None
            latest = self._latest
            return PlannerOutput(
                plan_version=int(latest.plan_version),
                source_frame_id=int(latest.source_frame_id),
                trajectory_world=np.asarray(latest.trajectory_world, dtype=np.float32).copy(),
                latency_ms=float(latest.latency_ms),
                successful_calls=int(latest.successful_calls),
                failed_calls=int(latest.failed_calls),
                last_error=str(latest.last_error),
            )

    def snapshot_status(self) -> tuple[int, int, str, float]:
        with self._lock:
            return (
                int(self._successful_calls),
                int(self._failed_calls),
                str(self._last_error),
                float(self._last_latency_ms),
            )

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            self._request_event.wait(timeout=0.05)
            self._request_event.clear()
            if self._stop_event.is_set():
                break
            with self._lock:
                pending = self._pending
                generation = int(self._generation)
                self._pending = None
            if pending is None:
                continue

            start_time = time.perf_counter()
            try:
                response = self._client.nogoal_step(
                    rgb_images=np.asarray([pending.rgb], dtype=np.uint8),
                    depth_images_m=np.asarray([pending.depth], dtype=np.float32),
                )
                traj_local = normalize_navdp_trajectory(response.trajectory)
                traj_world = trajectory_camera_to_world(
                    trajectory_local=traj_local,
                    camera_pos_world=pending.cam_pos,
                    camera_quat_wxyz=pending.cam_quat,
                    use_trajectory_z=self._use_trajectory_z,
                )
                if traj_world.shape[0] == 0:
                    raise ValueError("planner returned empty trajectory")

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                with self._lock:
                    if generation != self._generation:
                        continue
                    self._successful_calls += 1
                    self._plan_version += 1
                    self._last_error = ""
                    self._last_latency_ms = float(latency_ms)
                    self._latest = PlannerOutput(
                        plan_version=int(self._plan_version),
                        source_frame_id=int(pending.frame_id),
                        trajectory_world=traj_world.copy(),
                        latency_ms=float(latency_ms),
                        successful_calls=int(self._successful_calls),
                        failed_calls=int(self._failed_calls),
                        last_error="",
                    )
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                with self._lock:
                    if generation != self._generation:
                        continue
                    self._failed_calls += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._last_latency_ms = float(latency_ms)


class AsyncPointGoalPlanner:
    def __init__(self, client: NavDPPlannerClient, use_trajectory_z: bool, pointgoal_frame: Literal["camera", "robot"] = "camera"):
        self._client = client
        self._use_trajectory_z = bool(use_trajectory_z)
        frame = str(pointgoal_frame).strip().lower()
        if frame not in ("camera", "robot"):
            raise ValueError(f"unsupported pointgoal_frame: {pointgoal_frame}")
        self._pointgoal_frame: Literal["camera", "robot"] = frame
        self._lock = threading.Lock()
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: PlannerInput | None = None
        self._latest: PlannerOutput | None = None
        self._plan_version = -1
        self._successful_calls = 0
        self._failed_calls = 0
        self._last_error = ""
        self._last_latency_ms = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, name="navdp-pointgoal-planner", daemon=True)
        self._thread.start()

    def stop(self, timeout_sec: float = 2.0) -> None:
        self._stop_event.set()
        self._request_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(float(timeout_sec), 0.1))

    def submit(self, planner_input: PlannerInput) -> None:
        with self._lock:
            self._pending = planner_input
        self._request_event.set()

    def consume_latest(self, last_seen_version: int) -> PlannerOutput | None:
        with self._lock:
            if self._latest is None:
                return None
            if self._latest.plan_version <= int(last_seen_version):
                return None
            latest = self._latest
            return PlannerOutput(
                plan_version=int(latest.plan_version),
                source_frame_id=int(latest.source_frame_id),
                trajectory_world=np.asarray(latest.trajectory_world, dtype=np.float32).copy(),
                latency_ms=float(latest.latency_ms),
                successful_calls=int(latest.successful_calls),
                failed_calls=int(latest.failed_calls),
                last_error=str(latest.last_error),
            )

    def snapshot_status(self) -> tuple[int, int, str, float]:
        with self._lock:
            return (
                int(self._successful_calls),
                int(self._failed_calls),
                str(self._last_error),
                float(self._last_latency_ms),
            )

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            self._request_event.wait(timeout=0.05)
            self._request_event.clear()
            if self._stop_event.is_set():
                break
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                continue

            start_time = time.perf_counter()
            try:
                response = self._client.pointgoal_step(
                    point_goals=np.asarray([pending.local_goal_xy], dtype=np.float32),
                    rgb_images=np.asarray([pending.rgb], dtype=np.uint8),
                    depth_images_m=np.asarray([pending.depth], dtype=np.float32),
                    sensor_meta=pending.sensor_meta,
                )
                traj_local = normalize_navdp_trajectory(response.trajectory)
                traj_world = trajectory_local_to_world(
                    traj_local,
                    pointgoal_frame=self._pointgoal_frame,
                    camera_pos_world=pending.cam_pos,
                    camera_quat_wxyz=pending.cam_quat,
                    robot_pos_world=pending.robot_pos,
                    robot_yaw=pending.robot_yaw,
                    use_trajectory_z=self._use_trajectory_z,
                )
                if traj_world.shape[0] == 0:
                    raise ValueError("planner returned empty trajectory")

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                with self._lock:
                    self._successful_calls += 1
                    self._plan_version += 1
                    self._last_error = ""
                    self._last_latency_ms = float(latency_ms)
                    self._latest = PlannerOutput(
                        plan_version=int(self._plan_version),
                        source_frame_id=int(pending.frame_id),
                        trajectory_world=traj_world.copy(),
                        latency_ms=float(latency_ms),
                        successful_calls=int(self._successful_calls),
                        failed_calls=int(self._failed_calls),
                        last_error="",
                    )
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                with self._lock:
                    self._failed_calls += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._last_latency_ms = float(latency_ms)


class NavDPPlannerClient(Protocol):
    def pointgoal_step(
        self,
        point_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
        sensor_meta: dict[str, Any] | None = None,
    ):
        ...

    def nogoal_step(
        self,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
    ):
        ...
