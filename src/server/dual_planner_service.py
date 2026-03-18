from __future__ import annotations

import io
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

from common.cv2_compat import cv2
from common.geometry import normalize_navdp_trajectory, trajectory_camera_to_world
from inference.vlm import (
    System2Session,
    System2SessionConfig,
    System2SessionResult,
    build_vlm_endpoint,
)
from memory.models import MemoryContextBundle
from schemas.workers import (
    NavRequest,
    NavResult,
    S2Request,
    S2Result,
    build_error_result,
    finalize_worker_result,
    inherit_worker_metadata,
    stamp_worker_metadata,
)
from server.decision_engine import DecisionEngine


def parse_json_field(raw: str | None, fallback: Any) -> Any:
    if raw is None or raw == "":
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


@dataclass
class GoalCache:
    mode: str
    pixel_x: int | None
    pixel_y: int | None
    stop: bool
    yaw_delta_rad: float | None
    reason: str
    version: int
    updated_at: float
    source: str


@dataclass
class TrajectoryCache:
    trajectory_world: np.ndarray
    version: int
    goal_version: int
    updated_at: float
    latency_ms: float


class DualPlannerService:
    def __init__(self, args) -> None:
        self._lock = threading.Lock()
        self._decision_engine = DecisionEngine()
        self.navdp_url = str(args.navdp_url)
        self.vlm_endpoint = build_vlm_endpoint(str(args.vlm_url))
        self.vlm_model = str(args.vlm_model)
        self.vlm_temperature = float(args.vlm_temperature)
        self.vlm_top_k = int(args.vlm_top_k)
        self.vlm_top_p = float(args.vlm_top_p)
        self.vlm_min_p = float(args.vlm_min_p)
        self.vlm_repeat_penalty = float(args.vlm_repeat_penalty)
        self.s2_mode = str(args.s2_mode).lower()
        self.s1_period_sec = float(args.s1_period_sec)
        self.s2_period_sec = float(args.s2_period_sec)
        self.goal_ttl_sec = float(args.goal_ttl_sec)
        self.traj_ttl_sec = float(args.traj_ttl_sec)
        self.traj_max_stale_sec = float(args.traj_max_stale_sec)
        self.navdp_timeout_sec = float(args.navdp_timeout_sec)
        self.navdp_reset_timeout_sec = float(args.navdp_reset_timeout_sec)
        self.vlm_timeout_sec = float(args.vlm_timeout_sec)
        self.s2_failure_backoff_max_sec = float(args.s2_failure_backoff_max_sec)
        self.stop_threshold = float(args.stop_threshold)
        self.use_trajectory_z = bool(args.use_trajectory_z)
        self.debug_log = bool(args.debug_log)
        self.system2_session = System2Session(
            System2SessionConfig(
                endpoint=self.vlm_endpoint,
                model=self.vlm_model,
                temperature=self.vlm_temperature,
                top_k=self.vlm_top_k,
                top_p=self.vlm_top_p,
                min_p=self.vlm_min_p,
                repeat_penalty=self.vlm_repeat_penalty,
                timeout_sec=self.vlm_timeout_sec,
                num_history=int(getattr(args, "vlm_num_history", 8)),
                max_images_per_request=int(getattr(args, "vlm_max_images_per_request", 3)),
                mode="mock" if self.s2_mode == "mock" else "llm",
            )
        )
        self.initialized = False
        self.instruction = ""
        self.intrinsic: np.ndarray | None = None
        self.goal_cache: GoalCache | None = None
        self.traj_cache: TrajectoryCache | None = None
        self.goal_version = -1
        self.traj_version = -1
        self.last_s1_ts = 0.0
        self.last_s2_ts = 0.0
        self.s2_retry_after_ts = 0.0
        self.force_s2_pending = False
        self.last_s1_error = ""
        self.last_s2_error = ""
        self.s1_calls = 0
        self.s1_success = 0
        self.s1_fail = 0
        self.s2_calls = 0
        self.s2_success = 0
        self.s2_fail = 0
        self.step_calls = 0
        self.s1_inflight = False
        self.s2_inflight = False
        self.last_s2_reason = ""
        self.last_s2_raw_text = ""
        self.last_s2_requested_stop = False
        self.last_s2_effective_stop = False
        self.last_s2_mode = "wait"
        self.last_s2_history_frame_ids: tuple[int, ...] = ()
        self.last_s2_needs_requery = False
        self.s2_stop_suppressed_count = 0
        self._generation = 0
        self._active_task_id = ""

    def _url(self, base_url: str, path: str) -> str:
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    def _debug(self, text: str) -> None:
        if self.debug_log:
            print(f"[DUAL] {text}")

    def reset(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        intrinsic = np.asarray(payload.get("intrinsic"), dtype=np.float32)
        if intrinsic.size == 0:
            return False, {"error": "dual_reset requires intrinsic"}
        instruction = str(payload.get("instruction", self.instruction)).strip()
        if instruction == "":
            return False, {"error": "dual_reset requires non-empty instruction"}

        navdp_url = str(payload.get("navdp_url", self.navdp_url)).strip()
        if navdp_url == "":
            return False, {"error": "dual_reset requires navdp_url"}

        s1_period_sec = float(payload.get("s1_period_sec", self.s1_period_sec))
        s2_period_sec = float(payload.get("s2_period_sec", self.s2_period_sec))
        goal_ttl_sec = float(payload.get("goal_ttl_sec", self.goal_ttl_sec))
        traj_ttl_sec = float(payload.get("traj_ttl_sec", self.traj_ttl_sec))
        traj_max_stale_sec = float(payload.get("traj_max_stale_sec", self.traj_max_stale_sec))
        stop_threshold = float(payload.get("stop_threshold", self.stop_threshold))

        reset_payload = {
            "intrinsic": intrinsic.tolist(),
            "stop_threshold": stop_threshold,
            "batch_size": 1,
        }

        try:
            resp = requests.post(
                self._url(navdp_url, "navigator_reset"),
                json=reset_payload,
                timeout=float(self.navdp_reset_timeout_sec),
            )
            resp.raise_for_status()
            body = resp.json()
            algo = str(body.get("algo", ""))
            if algo == "":
                raise RuntimeError("navigator_reset returned empty algo")
        except Exception as exc:  # noqa: BLE001
            return False, {"error": f"navigator_reset failed: {type(exc).__name__}: {exc}"}

        with self._lock:
            self._generation += 1
            self._active_task_id = f"dual:{self._generation}"
            self.navdp_url = navdp_url
            self.s1_period_sec = max(s1_period_sec, 1.0e-3)
            self.s2_period_sec = max(s2_period_sec, 1.0e-3)
            self.goal_ttl_sec = max(goal_ttl_sec, 1.0e-3)
            self.traj_ttl_sec = max(traj_ttl_sec, 1.0e-3)
            self.traj_max_stale_sec = max(traj_max_stale_sec, self.traj_ttl_sec)
            self.stop_threshold = float(stop_threshold)
            self.instruction = instruction
            self.intrinsic = intrinsic.copy()
            self.goal_cache = None
            self.traj_cache = None
            self.goal_version = -1
            self.traj_version = -1
            self.last_s1_ts = 0.0
            self.last_s2_ts = 0.0
            self.s2_retry_after_ts = 0.0
            self.force_s2_pending = False
            self.last_s1_error = ""
            self.last_s2_error = ""
            self.s1_calls = 0
            self.s1_success = 0
            self.s1_fail = 0
            self.s2_calls = 0
            self.s2_success = 0
            self.s2_fail = 0
            self.step_calls = 0
            self.s1_inflight = False
            self.s2_inflight = False
            self.last_s2_reason = ""
            self.last_s2_raw_text = ""
            self.last_s2_requested_stop = False
            self.last_s2_effective_stop = False
            self.last_s2_mode = "wait"
            self.last_s2_history_frame_ids = ()
            self.last_s2_needs_requery = False
            self.s2_stop_suppressed_count = 0
            self.initialized = True
        self.system2_session.reset(instruction)

        self._debug(
            f"dual_reset navdp={self.navdp_url} s1={self.s1_period_sec:.3f}s "
            f"s2={self.s2_period_sec:.3f}s goal_ttl={self.goal_ttl_sec:.3f}s "
            f"traj_ttl={self.traj_ttl_sec:.3f}s"
        )
        return True, {"algo": "dual", "state": self.debug_state()}

    def _prepare_s2_request(
        self,
        *,
        step_id: int,
        events: dict[str, Any],
        memory_context: MemoryContextBundle | None,
    ) -> S2Request:
        prepared = self.system2_session.prepare_request(events=events, memory_context=memory_context)
        return S2Request(
            metadata=stamp_worker_metadata(
                source="dual_planner_service.s2",
                task_id=str(self._active_task_id),
                frame_id=int(prepared.frame_id if prepared is not None else step_id),
                timeout_ms=int(max(float(self.vlm_timeout_sec) * 1000.0, 0.0)),
            ),
            request=prepared,
            events=dict(events),
            memory_context=memory_context,
        )

    def _normalize_system2_result(self, result: System2SessionResult, *, request: S2Request) -> S2Result:
        metadata = inherit_worker_metadata(request.metadata, source=f"dual_planner_service.s2.{result.source}")
        if not result.ok or result.decision is None:
            return build_error_result(
                S2Result,
                metadata=metadata,
                error=str(result.error),
                source=str(result.source),
                latency_ms=float(result.latency_ms),
            )

        decision = result.decision
        pixel_x = None
        pixel_y = None
        stop = False
        yaw_delta_rad = None
        normalized_mode = str(decision.mode)
        if decision.mode == "pixel_goal" and decision.pixel_goal is not None:
            pixel_x = int(decision.pixel_goal[0])
            pixel_y = int(decision.pixel_goal[1])
        elif decision.mode == "stop":
            stop = True
        elif decision.mode == "yaw_left":
            normalized_mode = "yaw_delta"
            yaw_delta_rad = float(math.pi / 6.0)
        elif decision.mode == "yaw_right":
            normalized_mode = "yaw_delta"
            yaw_delta_rad = float(-math.pi / 6.0)
        else:
            normalized_mode = "wait"

        return S2Result(
            metadata=metadata,
            mode=normalized_mode,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            stop=bool(stop),
            yaw_delta_rad=yaw_delta_rad,
            reason=str(decision.reason),
            latency_ms=float(result.latency_ms),
            source=str(result.source),
            raw_text=str(decision.raw_text),
            history_frame_ids=tuple(decision.history_frame_ids),
            needs_requery=bool(decision.needs_requery),
        )

    def _call_s1(self, request: NavRequest) -> NavResult:
        start = time.perf_counter()
        metadata = inherit_worker_metadata(request.metadata, source="dual_planner_service.s1")
        if request.image_bgr is None or request.depth_m is None or request.pixel_goal is None:
            return build_error_result(
                NavResult,
                metadata=metadata,
                error="incomplete nav request",
            )
        depth = np.asarray(request.depth_m, dtype=np.float32)
        image = np.asarray(request.image_bgr, dtype=np.uint8)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        ok_img, img_buf = cv2.imencode(".jpg", image)
        if not ok_img:
            return build_error_result(NavResult, metadata=metadata, error="failed to encode rgb for NavDP")
        depth_mm_u16 = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
        ok_depth, depth_buf = cv2.imencode(".png", depth_mm_u16)
        if not ok_depth:
            return build_error_result(NavResult, metadata=metadata, error="failed to encode depth for NavDP")

        goal_data = {
            "goal_x": [int(request.pixel_goal[0])],
            "goal_y": [int(request.pixel_goal[1])],
            "sensor_meta": request.sensor_meta if isinstance(request.sensor_meta, dict) else {},
            "client_meta": {
                "dual_server": True,
                "rgb_shape": list(image.shape),
                "depth_shape": list(depth.shape),
            },
        }
        files = {
            "image": ("image.jpg", io.BytesIO(img_buf).getvalue(), "image/jpeg"),
            "depth": ("depth.png", io.BytesIO(depth_buf).getvalue(), "image/png"),
        }
        form_data = {"goal_data": json.dumps(goal_data)}
        try:
            resp = requests.post(
                self._url(self.navdp_url, "pixelgoal_step"),
                files=files,
                data=form_data,
                timeout=float(self.navdp_timeout_sec),
            )
            resp.raise_for_status()
            body = resp.json()
            trajectory_local = normalize_navdp_trajectory(np.asarray(body.get("trajectory"), dtype=np.float32))
            trajectory_world = trajectory_camera_to_world(
                trajectory_local=trajectory_local,
                camera_pos_world=np.asarray(request.cam_pos, dtype=np.float32),
                camera_quat_wxyz=np.asarray(request.cam_quat_wxyz, dtype=np.float32),
                use_trajectory_z=self.use_trajectory_z,
            )
            if trajectory_world.shape[0] == 0:
                raise ValueError("empty trajectory from NavDP")
            latency_ms = (time.perf_counter() - start) * 1000.0
            return NavResult(
                metadata=metadata,
                trajectory_world=trajectory_world,
                latency_ms=float(latency_ms),
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            return build_error_result(
                NavResult,
                metadata=metadata,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=float(latency_ms),
            )

    def _finish_s2(self, result: S2Result, finished_at: float, generation: int) -> None:
        with self._lock:
            if int(generation) != int(self._generation):
                return
            result = finalize_worker_result(
                result,
                task_id=str(self._active_task_id),
                now_ns=time.time_ns(),
            )
            self.s2_inflight = False
            self.s2_calls += 1
            self.last_s2_ts = float(finished_at)
            if result.ok:
                requested_stop = bool(result.stop or result.mode == "stop")
                stop_suppressed = requested_stop and self.traj_version < 0
                effective_mode = str(result.mode)
                effective_stop = requested_stop and not stop_suppressed
                effective_yaw_delta = result.yaw_delta_rad
                effective_pixel_x = result.pixel_x
                effective_pixel_y = result.pixel_y
                effective_needs_requery = bool(result.needs_requery)
                reason = str(result.reason)
                if stop_suppressed:
                    self.s2_stop_suppressed_count += 1
                    effective_mode = "wait"
                    effective_stop = False
                    effective_yaw_delta = None
                    effective_pixel_x = None
                    effective_pixel_y = None
                    effective_needs_requery = True
                    reason = reason + " [initial stop suppressed until first confirmed trajectory]"
                same_goal_as_current = (
                    self.goal_cache is not None
                    and str(self.goal_cache.mode) == str(effective_mode)
                    and self.goal_cache.pixel_x == effective_pixel_x
                    and self.goal_cache.pixel_y == effective_pixel_y
                    and bool(self.goal_cache.stop) == bool(effective_stop)
                    and (
                        (self.goal_cache.yaw_delta_rad is None and effective_yaw_delta is None)
                        or (
                            self.goal_cache.yaw_delta_rad is not None
                            and effective_yaw_delta is not None
                            and math.isclose(float(self.goal_cache.yaw_delta_rad), float(effective_yaw_delta), abs_tol=1.0e-6)
                        )
                    )
                )
                if same_goal_as_current and self.goal_cache is not None:
                    goal_version = int(self.goal_cache.version)
                else:
                    self.goal_version += 1
                    goal_version = int(self.goal_version)
                self.goal_cache = GoalCache(
                    mode=str(effective_mode),
                    pixel_x=None if effective_pixel_x is None else int(effective_pixel_x),
                    pixel_y=None if effective_pixel_y is None else int(effective_pixel_y),
                    stop=bool(effective_stop),
                    yaw_delta_rad=None if effective_yaw_delta is None else float(effective_yaw_delta),
                    reason=reason,
                    version=goal_version,
                    updated_at=float(finished_at),
                    source=str(result.source),
                )
                self.s2_success += 1
                self.last_s2_error = ""
                self.last_s2_reason = reason
                self.last_s2_raw_text = str(result.raw_text)
                self.last_s2_requested_stop = bool(requested_stop)
                self.last_s2_effective_stop = bool(effective_stop)
                self.last_s2_mode = str(effective_mode)
                self.last_s2_history_frame_ids = tuple(result.history_frame_ids)
                self.last_s2_needs_requery = bool(effective_needs_requery)
                self.s2_retry_after_ts = 0.0
                self.force_s2_pending = bool(effective_needs_requery)
            else:
                self.s2_fail += 1
                self.last_s2_error = str(result.error)
                self.last_s2_reason = ""
                self.last_s2_raw_text = str(result.raw_text)
                self.last_s2_requested_stop = False
                self.last_s2_effective_stop = False
                self.last_s2_mode = "wait"
                self.last_s2_history_frame_ids = ()
                self.last_s2_needs_requery = False
                base_delay = max(1.0, min(4.0, float(self.s2_period_sec)))
                delay = min(float(self.s2_failure_backoff_max_sec), base_delay * (2 ** max(self.s2_fail - 1, 0)))
                self.s2_retry_after_ts = float(finished_at) + delay
                self.force_s2_pending = True

    def _s2_worker(self, request: S2Request, generation: int) -> None:
        try:
            if request.request is None:
                raise RuntimeError("missing System2 request")
            session_result = self.system2_session.execute_request(request.request)
            if session_result.ok:
                self.system2_session.record_result(session_result)
            result = finalize_worker_result(
                self._normalize_system2_result(session_result, request=request),
                expected=request.metadata,
                now_ns=time.time_ns(),
            )
        except Exception as exc:  # noqa: BLE001
            result = build_error_result(
                S2Result,
                metadata=inherit_worker_metadata(request.metadata, source="dual_planner_service.s2.worker"),
                error=f"{type(exc).__name__}: {exc}",
                source="worker",
            )
        self._finish_s2(result=result, finished_at=time.time(), generation=int(generation))

    def _finish_s1(self, result: NavResult, goal_version: int, finished_at: float, generation: int) -> None:
        with self._lock:
            if int(generation) != int(self._generation):
                return
            self.s1_inflight = False
            self.s1_calls += 1
            self.last_s1_ts = float(finished_at)
            current_goal_version = self.goal_version
            result = finalize_worker_result(
                result,
                task_id=str(self._active_task_id),
                goal_version=int(current_goal_version),
                now_ns=time.time_ns(),
            )
            if not result.ok:
                self.s1_fail += 1
                self.last_s1_error = str(result.error or result.discard_reason or "unknown_s1_error")
                return
            if result.trajectory_world is not None:
                self.traj_version += 1
                self.traj_cache = TrajectoryCache(
                    trajectory_world=np.asarray(result.trajectory_world, dtype=np.float32).copy(),
                    version=int(self.traj_version),
                    goal_version=int(goal_version),
                    updated_at=float(finished_at),
                    latency_ms=float(result.latency_ms),
                )
                self.s1_success += 1
                self.last_s1_error = ""
            else:
                self.s1_fail += 1
                self.last_s1_error = str(result.error or result.discard_reason or "unknown_s1_error")

    def _s1_worker(self, request: NavRequest, generation: int) -> None:
        try:
            result = finalize_worker_result(
                self._call_s1(request),
                expected=request.metadata,
                now_ns=time.time_ns(),
            )
        except Exception as exc:  # noqa: BLE001
            result = build_error_result(
                NavResult,
                metadata=inherit_worker_metadata(request.metadata, source="dual_planner_service.s1.worker"),
                error=f"{type(exc).__name__}: {exc}",
            )
        self._finish_s1(
            result=result,
            goal_version=int(request.metadata.goal_version or -1),
            finished_at=time.time(),
            generation=int(generation),
        )

    def _read_state_snapshot(self) -> dict[str, Any]:
        with self._lock:
            goal = self.goal_cache
            traj = self.traj_cache
            now = time.time()
            goal_age = (now - goal.updated_at) if goal is not None else None
            traj_age = (now - traj.updated_at) if traj is not None else None
            return {
                "initialized": bool(self.initialized),
                "instruction": self.instruction,
                "navdp_url": self.navdp_url,
                "vlm_endpoint": self.vlm_endpoint,
                "s2_mode": self.s2_mode,
                "periods": {
                    "s1_period_sec": self.s1_period_sec,
                    "s2_period_sec": self.s2_period_sec,
                    "goal_ttl_sec": self.goal_ttl_sec,
                    "traj_ttl_sec": self.traj_ttl_sec,
                    "traj_max_stale_sec": self.traj_max_stale_sec,
                },
                "goal_cache": None
                if goal is None
                else {
                    "mode": goal.mode,
                    "pixel_x": goal.pixel_x,
                    "pixel_y": goal.pixel_y,
                    "stop": goal.stop,
                    "yaw_delta_rad": goal.yaw_delta_rad,
                    "reason": goal.reason,
                    "version": goal.version,
                    "age_sec": goal_age,
                    "source": goal.source,
                },
                "traj_cache": None
                if traj is None
                else {
                    "version": traj.version,
                    "goal_version": traj.goal_version,
                    "length": int(traj.trajectory_world.shape[0]),
                    "age_sec": traj_age,
                    "latency_ms": traj.latency_ms,
                },
                "stats": {
                    "step_calls": self.step_calls,
                    "s1_calls": self.s1_calls,
                    "s1_success": self.s1_success,
                    "s1_fail": self.s1_fail,
                    "s2_calls": self.s2_calls,
                    "s2_success": self.s2_success,
                    "s2_fail": self.s2_fail,
                    "last_s1_error": self.last_s1_error,
                    "last_s2_error": self.last_s2_error,
                    "last_s2_reason": self.last_s2_reason,
                    "last_s2_requested_stop": self.last_s2_requested_stop,
                    "last_s2_effective_stop": self.last_s2_effective_stop,
                    "last_s2_mode": self.last_s2_mode,
                    "last_s2_history_frame_ids": list(self.last_s2_history_frame_ids),
                    "last_s2_needs_requery": self.last_s2_needs_requery,
                    "last_s2_raw_text": self.last_s2_raw_text[:400],
                    "s2_stop_suppressed_count": self.s2_stop_suppressed_count,
                    "s1_inflight": self.s1_inflight,
                    "s2_inflight": self.s2_inflight,
                    "force_s2_pending": self.force_s2_pending,
                    "s2_retry_after_ts": self.s2_retry_after_ts,
                },
                "system2_session": self.system2_session.debug_state(),
            }

    def debug_state(self) -> dict[str, Any]:
        return self._read_state_snapshot()

    def step(
        self,
        *,
        image_bgr: np.ndarray,
        depth_m: np.ndarray,
        step_id: int,
        cam_pos: np.ndarray,
        cam_quat_wxyz: np.ndarray,
        sensor_meta: dict[str, Any],
        memory_context: MemoryContextBundle | None,
        events: dict[str, Any],
    ) -> dict[str, Any]:
        image = np.asarray(image_bgr, dtype=np.uint8)
        depth = np.asarray(depth_m, dtype=np.float32)
        self.system2_session.observe(int(step_id), image)
        now = time.time()
        launch_s2 = False
        launch_s1 = False
        s2_request: S2Request | None = None
        s1_request: NavRequest | None = None
        s1_goal_version = -1
        s1_pixel_goal = (0, 0)
        generation = -1

        with self._lock:
            if not self.initialized:
                raise RuntimeError("dual_reset must be called before dual_step")
            generation = int(self._generation)
            self.step_calls += 1
            goal = self.goal_cache
            traj = self.traj_cache
            directive = self._decision_engine.evaluate_dual(
                now=now,
                goal_cache=goal,
                traj_cache=traj,
                last_s1_ts=self.last_s1_ts,
                last_s2_ts=self.last_s2_ts,
                s1_period_sec=self.s1_period_sec,
                s2_period_sec=self.s2_period_sec,
                goal_ttl_sec=self.goal_ttl_sec,
                traj_ttl_sec=self.traj_ttl_sec,
                traj_max_stale_sec=self.traj_max_stale_sec,
                s2_retry_after_ts=self.s2_retry_after_ts,
                force_s2_pending=self.force_s2_pending,
                events=events,
            )
            force_s2 = bool(directive.force_s2)
            should_s2 = bool(directive.launch_s2)
            if should_s2 and not self.s2_inflight:
                self.s2_inflight = True
                launch_s2 = True

            goal = self.goal_cache
            traj = self.traj_cache
            should_s1 = bool(directive.launch_s1)
            if should_s1 and not self.s1_inflight and goal is not None:
                self.s1_inflight = True
                launch_s1 = True
                s1_goal_version = int(goal.version)
                s1_pixel_goal = (int(goal.pixel_x), int(goal.pixel_y))

        if launch_s2:
            try:
                s2_request = self._prepare_s2_request(
                    step_id=int(step_id),
                    events=dict(events),
                    memory_context=memory_context,
                )
            except Exception as exc:  # noqa: BLE001
                self._finish_s2(
                    build_error_result(
                        S2Result,
                        metadata=stamp_worker_metadata(
                            source="dual_planner_service.s2.prepare",
                            task_id=str(self._active_task_id),
                            frame_id=int(step_id),
                            timeout_ms=int(max(float(self.vlm_timeout_sec) * 1000.0, 0.0)),
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                        source="prepare_request",
                    ),
                    finished_at=time.time(),
                    generation=int(generation),
                )
                launch_s2 = False

        if launch_s1:
            s1_request = NavRequest(
                metadata=stamp_worker_metadata(
                    source="dual_planner_service.s1",
                    task_id=str(self._active_task_id),
                    frame_id=int(step_id),
                    timeout_ms=int(max(float(self.navdp_timeout_sec) * 1000.0, 0.0)),
                    goal_version=int(s1_goal_version),
                ),
                image_bgr=image.copy(),
                depth_m=depth.copy(),
                pixel_goal=(int(s1_pixel_goal[0]), int(s1_pixel_goal[1])),
                sensor_meta=dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
                cam_pos=np.asarray(cam_pos, dtype=np.float32).copy(),
                cam_quat_wxyz=np.asarray(cam_quat_wxyz, dtype=np.float32).copy(),
            )

        if launch_s2 and s2_request is not None:
            threading.Thread(
                target=self._s2_worker,
                args=(s2_request, int(generation)),
                name="dual-s2-worker",
                daemon=True,
            ).start()

        if launch_s1 and s1_request is not None:
            threading.Thread(
                target=self._s1_worker,
                args=(s1_request, int(generation)),
                name="dual-s1-worker",
                daemon=True,
            ).start()

        with self._lock:
            goal = self.goal_cache
            traj = self.traj_cache
            planner_mode = "wait" if goal is None else "trajectory"
            planner_yaw_delta_rad = None
            planner_reason = ""
            pixel_goal = None
            stop = bool(goal.stop) if goal is not None else False
            goal_version = int(goal.version) if goal is not None else -1
            traj_version = int(traj.version) if traj is not None else -1
            if goal is not None:
                planner_reason = str(goal.reason)
                if goal.mode == "pixel_goal" and goal.pixel_x is not None and goal.pixel_y is not None:
                    pixel_goal = [int(goal.pixel_x), int(goal.pixel_y)]
                    planner_mode = "trajectory"
                elif goal.mode == "yaw_delta":
                    planner_mode = "yaw_delta"
                    planner_yaw_delta_rad = None if goal.yaw_delta_rad is None else float(goal.yaw_delta_rad)
                elif goal.mode == "stop":
                    planner_mode = "stop"
                else:
                    planner_mode = "wait"

            used_cached_traj = False
            if traj is None or planner_mode != "trajectory":
                traj_age = float("inf")
                trajectory_world = np.zeros((0, 3), dtype=np.float32)
            else:
                traj_age = max(0.0, now - traj.updated_at)
                if stop:
                    trajectory_world = np.zeros((0, 3), dtype=np.float32)
                elif traj_age <= self.traj_max_stale_sec:
                    trajectory_world = traj.trajectory_world.copy()
                    used_cached_traj = True
                else:
                    trajectory_world = np.zeros((0, 3), dtype=np.float32)
                    self.force_s2_pending = True

            stale_sec = -1.0 if not math.isfinite(traj_age) else float(traj_age)
            debug = {
                "step_id": int(step_id),
                "called_s2": bool(launch_s2),
                "called_s1": bool(launch_s1),
                "s2_ok": None,
                "s1_ok": None,
                "s2_error": self.last_s2_error,
                "s1_error": self.last_s1_error,
                "planner_control_mode": planner_mode,
                "planner_control_reason": planner_reason,
                "planner_control_yaw_delta_rad": planner_yaw_delta_rad,
                "s2_source": "",
                "s2_latency_ms": 0.0,
                "s1_latency_ms": 0.0,
                "goal_age_sec": None if goal is None else (now - goal.updated_at),
                "traj_age_sec": None if traj is None else traj_age,
                "low_confidence_traj": bool(traj is not None and self.traj_ttl_sec < traj_age <= self.traj_max_stale_sec),
                "stale_drop": bool(traj is not None and traj_age > self.traj_max_stale_sec),
                "force_s2_pending": self.force_s2_pending,
                "s1_inflight": self.s1_inflight,
                "s2_inflight": self.s2_inflight,
                "stats": {
                    "step_calls": self.step_calls,
                    "s1_calls": self.s1_calls,
                    "s1_success": self.s1_success,
                    "s1_fail": self.s1_fail,
                    "s2_calls": self.s2_calls,
                    "s2_success": self.s2_success,
                    "s2_fail": self.s2_fail,
                },
            }

        return {
            "trajectory_world": trajectory_world.tolist(),
            "pixel_goal": pixel_goal,
            "stop": stop,
            "goal_version": goal_version,
            "traj_version": traj_version,
            "used_cached_traj": used_cached_traj,
            "stale_sec": float(stale_sec),
            "planner_control": {
                "mode": planner_mode,
                "yaw_delta_rad": planner_yaw_delta_rad,
                "reason": planner_reason,
            },
            "debug": debug,
        }
