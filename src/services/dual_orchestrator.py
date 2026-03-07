from __future__ import annotations

import base64
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

S2_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pixel_x": {"type": "integer"},
        "pixel_y": {"type": "integer"},
        "stop": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["pixel_x", "pixel_y", "stop", "reason"],
    "additionalProperties": False,
}


def build_vlm_endpoint(url: str) -> str:
    raw = str(url).strip()
    if raw == "":
        return "http://127.0.0.1:8080/v1/chat/completions"
    if raw.endswith("/v1/chat/completions"):
        return raw
    return raw.rstrip("/") + "/v1/chat/completions"


def parse_json_field(raw: str | None, fallback: Any) -> Any:
    if raw is None or raw == "":
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in VLM output.")
    return json.loads(text[start : end + 1])


def extract_chat_content(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise ValueError("VLM response missing choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)
        return "\n".join(parts)
    raise ValueError("Unsupported VLM content format.")


@dataclass
class GoalCache:
    pixel_x: int
    pixel_y: int
    stop: bool
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


@dataclass
class S2Result:
    ok: bool
    pixel_x: int = 0
    pixel_y: int = 0
    stop: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    source: str = "none"
    error: str = ""
    raw_text: str = ""


@dataclass
class S1Result:
    ok: bool
    trajectory_world: np.ndarray | None = None
    latency_ms: float = 0.0
    error: str = ""


class DualOrchestrator:
    def __init__(self, args) -> None:
        self._lock = threading.Lock()
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
        self.s2_stop_suppressed_count = 0
        self._generation = 0

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
            self.s2_stop_suppressed_count = 0
            self.initialized = True

        self._debug(
            f"dual_reset navdp={self.navdp_url} s1={self.s1_period_sec:.3f}s "
            f"s2={self.s2_period_sec:.3f}s goal_ttl={self.goal_ttl_sec:.3f}s "
            f"traj_ttl={self.traj_ttl_sec:.3f}s"
        )
        return True, {"algo": "dual", "state": self.debug_state()}

    def _call_s2_mock(self, image_bgr: np.ndarray, events: dict[str, Any]) -> S2Result:
        h, w = image_bgr.shape[:2]
        collision_risk = bool(events.get("collision_risk", False))
        stuck = bool(events.get("stuck", False))
        if collision_risk:
            pixel_x = int(0.25 * w)
            pixel_y = int(0.70 * h)
            reason = "mock_collision_avoid"
        elif stuck:
            pixel_x = int(0.75 * w)
            pixel_y = int(0.65 * h)
            reason = "mock_stuck_recover"
        else:
            pixel_x = int(0.50 * w)
            pixel_y = int(0.70 * h)
            reason = "mock_forward"
        return S2Result(
            ok=True,
            pixel_x=int(np.clip(pixel_x, 0, max(w - 1, 0))),
            pixel_y=int(np.clip(pixel_y, 0, max(h - 1, 0))),
            stop=False,
            reason=reason,
            source="mock",
            latency_ms=0.0,
        )

    def _call_s2_vlm(self, image_bgr: np.ndarray, events: dict[str, Any]) -> S2Result:
        start = time.perf_counter()
        ok_jpg, jpg = cv2.imencode(".jpg", image_bgr)
        if not ok_jpg:
            return S2Result(ok=False, error="failed to encode image for VLM")
        jpg_base64 = base64.b64encode(io.BytesIO(jpg).getvalue()).decode("ascii")
        h, w = image_bgr.shape[:2]
        prompt = (
            "You are System2 of a navigation dual-system.\n"
            "Return ONLY JSON object with fields: "
            "{\"pixel_x\":int,\"pixel_y\":int,\"stop\":bool,\"reason\":string}.\n"
            f"Image size: width={w}, height={h}.\n"
            f"Instruction: {self.instruction}\n"
            f"Events: {json.dumps(events, ensure_ascii=True)}.\n"
            "Pick pixel near free-space direction. Clamp bounds.\n"
            "Set stop=true ONLY if the robot has already reached the destination and should remain stopped now.\n"
            "If any further navigation is needed, or if the scene is ambiguous, set stop=false."
        )
        payload = {
            "model": self.vlm_model,
            "temperature": float(self.vlm_temperature),
            "top_k": int(self.vlm_top_k),
            "top_p": float(self.vlm_top_p),
            "min_p": float(self.vlm_min_p),
            "repeat_penalty": float(self.vlm_repeat_penalty),
            "max_tokens": 96,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": S2_JSON_SCHEMA},
            },
            "messages": [
                {
                    "role": "system",
                    "content": "Strictly output valid JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64}"}},
                    ],
                },
            ],
        }
        try:
            resp = requests.post(self.vlm_endpoint, json=payload, timeout=float(self.vlm_timeout_sec))
            resp.raise_for_status()
            body = resp.json()
            text = extract_chat_content(body)
            parsed = extract_json_object(text)
            raw_x = int(parsed.get("pixel_x", parsed.get("x", w // 2)))
            raw_y = int(parsed.get("pixel_y", parsed.get("y", int(0.7 * h))))
            stop = bool(parsed.get("stop", False))
            reason = str(parsed.get("reason", "llm"))
            latency_ms = (time.perf_counter() - start) * 1000.0
            return S2Result(
                ok=True,
                pixel_x=int(np.clip(raw_x, 0, max(w - 1, 0))),
                pixel_y=int(np.clip(raw_y, 0, max(h - 1, 0))),
                stop=stop,
                reason=reason,
                source="llm",
                raw_text=text,
                latency_ms=float(latency_ms),
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            return S2Result(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                source="llm",
                latency_ms=float(latency_ms),
            )

    def _call_s2(self, image_bgr: np.ndarray, events: dict[str, Any]) -> S2Result:
        if self.s2_mode == "mock":
            return self._call_s2_mock(image_bgr, events)
        return self._call_s2_vlm(image_bgr, events)

    def _call_s1(
        self,
        image_bgr: np.ndarray,
        depth_m: np.ndarray,
        pixel_goal: tuple[int, int],
        sensor_meta: dict[str, Any],
        cam_pos: np.ndarray,
        cam_quat_wxyz: np.ndarray,
    ) -> S1Result:
        start = time.perf_counter()
        depth = np.asarray(depth_m, dtype=np.float32)
        image = np.asarray(image_bgr, dtype=np.uint8)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        ok_img, img_buf = cv2.imencode(".jpg", image)
        if not ok_img:
            return S1Result(ok=False, error="failed to encode rgb for NavDP")
        depth_mm_u16 = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
        ok_depth, depth_buf = cv2.imencode(".png", depth_mm_u16)
        if not ok_depth:
            return S1Result(ok=False, error="failed to encode depth for NavDP")

        goal_data = {
            "goal_x": [int(pixel_goal[0])],
            "goal_y": [int(pixel_goal[1])],
            "sensor_meta": sensor_meta if isinstance(sensor_meta, dict) else {},
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
                camera_pos_world=np.asarray(cam_pos, dtype=np.float32),
                camera_quat_wxyz=np.asarray(cam_quat_wxyz, dtype=np.float32),
                use_trajectory_z=self.use_trajectory_z,
            )
            if trajectory_world.shape[0] == 0:
                raise ValueError("empty trajectory from NavDP")
            latency_ms = (time.perf_counter() - start) * 1000.0
            return S1Result(ok=True, trajectory_world=trajectory_world, latency_ms=float(latency_ms))
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            return S1Result(ok=False, error=f"{type(exc).__name__}: {exc}", latency_ms=float(latency_ms))

    def _finish_s2(self, result: S2Result, finished_at: float, generation: int) -> None:
        with self._lock:
            if int(generation) != int(self._generation):
                return
            self.s2_inflight = False
            self.s2_calls += 1
            self.last_s2_ts = float(finished_at)
            if result.ok:
                requested_stop = bool(result.stop)
                stop_suppressed = requested_stop and self.traj_version < 0
                effective_stop = requested_stop and not stop_suppressed
                reason = str(result.reason)
                if stop_suppressed:
                    self.s2_stop_suppressed_count += 1
                    reason = reason + " [initial stop suppressed until first confirmed trajectory]"
                same_goal_as_current = (
                    self.goal_cache is not None
                    and int(self.goal_cache.pixel_x) == int(result.pixel_x)
                    and int(self.goal_cache.pixel_y) == int(result.pixel_y)
                    and bool(self.goal_cache.stop) == bool(effective_stop)
                )
                if same_goal_as_current and self.goal_cache is not None:
                    goal_version = int(self.goal_cache.version)
                else:
                    self.goal_version += 1
                    goal_version = int(self.goal_version)
                self.goal_cache = GoalCache(
                    pixel_x=int(result.pixel_x),
                    pixel_y=int(result.pixel_y),
                    stop=bool(effective_stop),
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
                self.s2_retry_after_ts = 0.0
                self.force_s2_pending = False
            else:
                self.s2_fail += 1
                self.last_s2_error = str(result.error)
                self.last_s2_reason = ""
                self.last_s2_raw_text = str(result.raw_text)
                self.last_s2_requested_stop = False
                self.last_s2_effective_stop = False
                base_delay = max(1.0, min(4.0, float(self.s2_period_sec)))
                delay = min(float(self.s2_failure_backoff_max_sec), base_delay * (2 ** max(self.s2_fail - 1, 0)))
                self.s2_retry_after_ts = float(finished_at) + delay
                self.force_s2_pending = True

    def _s2_worker(self, image_bgr: np.ndarray, events: dict[str, Any], generation: int) -> None:
        try:
            result = self._call_s2(image_bgr=image_bgr, events=events)
        except Exception as exc:  # noqa: BLE001
            result = S2Result(ok=False, error=f"{type(exc).__name__}: {exc}", source="worker")
        self._finish_s2(result=result, finished_at=time.time(), generation=int(generation))

    def _finish_s1(self, result: S1Result, goal_version: int, finished_at: float, generation: int) -> None:
        with self._lock:
            if int(generation) != int(self._generation):
                return
            self.s1_inflight = False
            self.s1_calls += 1
            self.last_s1_ts = float(finished_at)
            current_goal_version = self.goal_version
            if goal_version != current_goal_version:
                self.last_s1_error = (
                    f"dropped_stale_plan goal_version={goal_version} current_goal_version={current_goal_version}"
                )
                return
            if result.ok and result.trajectory_world is not None:
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
                self.last_s1_error = result.error if result is not None else "unknown_s1_error"

    def _s1_worker(
        self,
        image_bgr: np.ndarray,
        depth_m: np.ndarray,
        pixel_goal: tuple[int, int],
        sensor_meta: dict[str, Any],
        cam_pos: np.ndarray,
        cam_quat_wxyz: np.ndarray,
        goal_version: int,
        generation: int,
    ) -> None:
        try:
            result = self._call_s1(
                image_bgr=image_bgr,
                depth_m=depth_m,
                pixel_goal=pixel_goal,
                sensor_meta=sensor_meta,
                cam_pos=cam_pos,
                cam_quat_wxyz=cam_quat_wxyz,
            )
        except Exception as exc:  # noqa: BLE001
            result = S1Result(ok=False, error=f"{type(exc).__name__}: {exc}")
        self._finish_s1(
            result=result,
            goal_version=int(goal_version),
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
                    "pixel_x": goal.pixel_x,
                    "pixel_y": goal.pixel_y,
                    "stop": goal.stop,
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
                    "last_s2_raw_text": self.last_s2_raw_text[:400],
                    "s2_stop_suppressed_count": self.s2_stop_suppressed_count,
                    "s1_inflight": self.s1_inflight,
                    "s2_inflight": self.s2_inflight,
                    "force_s2_pending": self.force_s2_pending,
                    "s2_retry_after_ts": self.s2_retry_after_ts,
                },
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
        events: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        launch_s2 = False
        launch_s1 = False
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
            external_force = bool(events.get("force_s2", False)) or bool(events.get("stuck", False)) or bool(
                events.get("collision_risk", False)
            )
            force_s2 = bool(external_force or self.force_s2_pending)
            goal_missing = goal is None
            goal_stale = (goal is not None) and ((now - goal.updated_at) > self.goal_ttl_sec)
            due_s2 = (now - self.last_s2_ts) >= self.s2_period_sec
            awaiting_first_traj = goal is not None and traj is None and not goal.stop
            backoff_active = now < self.s2_retry_after_ts
            should_s2 = force_s2 or goal_missing or goal_stale or due_s2
            if awaiting_first_traj and not force_s2:
                # Keep the first S2 goal stable long enough for S1 to produce an initial trajectory.
                should_s2 = goal_missing
            if backoff_active and not force_s2:
                should_s2 = False
            if should_s2 and not self.s2_inflight:
                self.s2_inflight = True
                launch_s2 = True

            goal = self.goal_cache
            traj = self.traj_cache
            goal_changed = goal is not None and (traj is None or traj.goal_version != goal.version)
            traj_missing = traj is None
            traj_stale = (traj is not None) and ((now - traj.updated_at) > self.traj_ttl_sec)
            due_s1 = (now - self.last_s1_ts) >= self.s1_period_sec
            should_s1 = (goal is not None and not goal.stop) and (goal_changed or traj_missing or traj_stale or due_s1)
            if should_s1 and not self.s1_inflight and goal is not None:
                self.s1_inflight = True
                launch_s1 = True
                s1_goal_version = int(goal.version)
                s1_pixel_goal = (int(goal.pixel_x), int(goal.pixel_y))

        if launch_s2:
            threading.Thread(
                target=self._s2_worker,
                args=(np.asarray(image_bgr, dtype=np.uint8).copy(), dict(events), int(generation)),
                name="dual-s2-worker",
                daemon=True,
            ).start()

        if launch_s1:
            threading.Thread(
                target=self._s1_worker,
                args=(
                    np.asarray(image_bgr, dtype=np.uint8).copy(),
                    np.asarray(depth_m, dtype=np.float32).copy(),
                    s1_pixel_goal,
                    dict(sensor_meta) if isinstance(sensor_meta, dict) else {},
                    np.asarray(cam_pos, dtype=np.float32).copy(),
                    np.asarray(cam_quat_wxyz, dtype=np.float32).copy(),
                    int(s1_goal_version),
                    int(generation),
                ),
                name="dual-s1-worker",
                daemon=True,
            ).start()

        with self._lock:
            goal = self.goal_cache
            traj = self.traj_cache
            stop = bool(goal.stop) if goal is not None else False
            goal_version = int(goal.version) if goal is not None else -1
            traj_version = int(traj.version) if traj is not None else -1
            pixel_goal = [int(goal.pixel_x), int(goal.pixel_y)] if goal is not None else None

            if traj is None:
                traj_age = float("inf")
                trajectory_world = np.zeros((0, 3), dtype=np.float32)
            else:
                traj_age = max(0.0, now - traj.updated_at)
                if stop:
                    trajectory_world = np.zeros((0, 3), dtype=np.float32)
                elif traj_age <= self.traj_max_stale_sec:
                    trajectory_world = traj.trajectory_world.copy()
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
            "used_cached_traj": True,
            "stale_sec": float(stale_sec),
            "debug": debug,
        }
