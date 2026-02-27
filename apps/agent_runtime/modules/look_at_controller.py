from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple


@dataclass
class LookAtParams:
    max_rate_hz: float = 20.0
    deadband_px: float = 8.0
    timeout_sec: float = 2.5
    grace_sec: float = 0.5
    smoothing: float = 0.45
    fallback_behavior: str = "hold"
    kx: float = 0.0018
    ky: float = 0.0016
    max_rate_deg_s: float = 30.0
    target_max_age_s: float = 0.6

    @property
    def max_rate_rad_s(self) -> float:
        return math.radians(float(self.max_rate_deg_s))

    @classmethod
    def from_overrides(cls, overrides: Optional[dict]) -> "LookAtParams":
        if not overrides:
            return cls()
        obj = cls()
        if "max_rate_hz" in overrides:
            obj.max_rate_hz = max(1.0, float(overrides["max_rate_hz"]))
        if "deadband_px" in overrides:
            obj.deadband_px = max(0.0, float(overrides["deadband_px"]))
        if "timeout_sec" in overrides:
            obj.timeout_sec = max(0.2, float(overrides["timeout_sec"]))
        if "grace_sec" in overrides:
            obj.grace_sec = max(0.0, float(overrides["grace_sec"]))
        if "smoothing" in overrides:
            obj.smoothing = max(0.0, min(1.0, float(overrides["smoothing"])))
        if "fallback_behavior" in overrides:
            obj.fallback_behavior = str(overrides["fallback_behavior"]).strip().lower() or obj.fallback_behavior
        if "kx" in overrides:
            obj.kx = float(overrides["kx"])
        if "ky" in overrides:
            obj.ky = float(overrides["ky"])
        if "max_rate_deg_s" in overrides:
            obj.max_rate_deg_s = max(1.0, float(overrides["max_rate_deg_s"]))
        if "target_max_age_s" in overrides:
            obj.target_max_age_s = max(0.05, float(overrides["target_max_age_s"]))
        return obj


@dataclass
class LookAtStatus:
    state: str = "idle"
    object_label: str = ""
    target_score: float = 0.0
    detections_ok: bool = False
    message: str = ""
    updated_at: float = field(default_factory=lambda: time.time())


class LookAtController:
    """Persistent image-space P-controller for object-centric camera orientation."""

    def __init__(
        self,
        get_target: Callable[[str, float], Optional[Any]],
        get_camera_aim: Callable[[], Tuple[float, float]],
        command_camera_aim: Callable[[float, float, str], Tuple[float, float]],
    ) -> None:
        self._get_target = get_target
        self._get_camera_aim = get_camera_aim
        self._command_camera_aim = command_camera_aim

        self._params = LookAtParams()
        self._object_label = ""
        self._status = LookAtStatus()
        self._last_step_mono = time.monotonic()
        self._lost_since_mono: Optional[float] = None
        self._yaw_rad = 0.0
        self._pitch_rad = 0.0

    @property
    def params(self) -> LookAtParams:
        return self._params

    def get_status(self) -> LookAtStatus:
        return LookAtStatus(
            state=self._status.state,
            object_label=self._status.object_label,
            target_score=self._status.target_score,
            detections_ok=self._status.detections_ok,
            message=self._status.message,
            updated_at=self._status.updated_at,
        )

    def activate(self, object_label: str, overrides: Optional[dict] = None) -> LookAtStatus:
        label = str(object_label or "").strip().lower()
        if not label:
            return self.stop(reason="stopped")

        self._params = LookAtParams.from_overrides(overrides)
        self._object_label = label
        self._lost_since_mono = None
        self._last_step_mono = time.monotonic()
        self._yaw_rad, self._pitch_rad = self._get_camera_aim()
        self._status = LookAtStatus(
            state="tracking",
            object_label=label,
            target_score=0.0,
            detections_ok=False,
            message="look_at activated",
            updated_at=time.time(),
        )
        return self.get_status()

    def stop(self, reason: str = "stopped") -> LookAtStatus:
        self._object_label = ""
        self._lost_since_mono = None
        self._status = LookAtStatus(
            state=str(reason or "stopped"),
            object_label="",
            target_score=float(self._status.target_score),
            detections_ok=False,
            message=str(reason or "stopped"),
            updated_at=time.time(),
        )
        return self.get_status()

    def step(self) -> LookAtStatus:
        now_mono = time.monotonic()
        dt = max(1e-3, now_mono - self._last_step_mono)
        self._last_step_mono = now_mono

        if not self._object_label:
            self._status.object_label = ""
            self._status.state = "idle" if self._status.state in {"idle", "stopped"} else self._status.state
            self._status.updated_at = time.time()
            return self.get_status()

        target = self._get_target(self._object_label, self._params.target_max_age_s)
        if target is None:
            if self._lost_since_mono is None:
                self._lost_since_mono = now_mono
            lost_elapsed = now_mono - self._lost_since_mono

            if self._params.fallback_behavior == "hold" and lost_elapsed <= self._params.grace_sec:
                self._yaw_rad, self._pitch_rad = self._command_camera_aim(
                    self._yaw_rad,
                    self._pitch_rad,
                    "look_at_hold",
                )
                self._status.state = "tracking"
                self._status.message = "target missing, holding last direction"
                self._status.detections_ok = False
                self._status.updated_at = time.time()
                return self.get_status()

            if lost_elapsed >= self._params.timeout_sec:
                self._object_label = ""
                self._status.state = "target_lost"
                self._status.object_label = ""
                self._status.message = "target lost timeout"
                self._status.detections_ok = False
                self._status.updated_at = time.time()
                return self.get_status()

            self._status.state = "target_lost"
            self._status.message = "target not found"
            self._status.detections_ok = False
            self._status.updated_at = time.time()
            return self.get_status()

        self._lost_since_mono = None
        cx = float(getattr(target, "cx", 0.0))
        cy = float(getattr(target, "cy", 0.0))
        score = float(getattr(target, "score", 0.0))
        width = max(1.0, float(getattr(target, "image_width", 0.0)))
        height = max(1.0, float(getattr(target, "image_height", 0.0)))

        ex = cx - 0.5 * width
        ey = cy - 0.5 * height
        if abs(ex) < self._params.deadband_px:
            ex = 0.0
        if abs(ey) < self._params.deadband_px:
            ey = 0.0

        yaw_delta = -self._params.kx * ex
        pitch_delta = -self._params.ky * ey
        max_step = self._params.max_rate_rad_s * dt
        yaw_delta = max(-max_step, min(max_step, yaw_delta))
        pitch_delta = max(-max_step, min(max_step, pitch_delta))

        target_yaw = self._yaw_rad + yaw_delta
        target_pitch = self._pitch_rad + pitch_delta
        alpha = self._params.smoothing
        cmd_yaw = (1.0 - alpha) * self._yaw_rad + alpha * target_yaw
        cmd_pitch = (1.0 - alpha) * self._pitch_rad + alpha * target_pitch
        self._yaw_rad, self._pitch_rad = self._command_camera_aim(cmd_yaw, cmd_pitch, "look_at")

        self._status.state = "tracking"
        self._status.object_label = self._object_label
        self._status.target_score = score
        self._status.detections_ok = True
        self._status.message = "tracking"
        self._status.updated_at = time.time()
        return self.get_status()
