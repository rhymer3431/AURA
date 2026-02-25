from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from typing import Awaitable, Callable, Dict, List, Optional

from .contracts import Pose2D, SlamMode


class SLAMMonitor:
    """Tracks localization confidence and switches mapping/localization mode."""

    def __init__(self, cfg: Dict) -> None:
        self.t_low = float(cfg.get("t_low", 0.35))
        self.t_high = float(cfg.get("t_high", 0.70))
        self.recover_hold_s = float(cfg.get("recover_hold_s", 4.0))
        self.covariance_scale = float(cfg.get("covariance_scale", 2.5))
        self.mock_mode = bool(cfg.get("mock_mode", True))
        self.mock_cycle_s = float(cfg.get("mock_cycle_s", 40.0))

        self.c_loc = 1.0
        self.mode = SlamMode.LOCALIZATION
        self.latest_pose = Pose2D(0.0, 0.0, 0.0, frame_id="map", covariance_norm=0.0)

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._above_high_since: Optional[float] = None
        self._mode_callbacks: List[Callable[[str, float], Awaitable[None] | None]] = []

    def register_mode_callback(self, callback: Callable[[str, float], Awaitable[None] | None]) -> None:
        self._mode_callbacks.append(callback)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self.mock_mode:
            self._task = asyncio.create_task(self._mock_loop(), name="slam_monitor")
            logging.info("SLAM monitor started in mock mode.")
        else:
            logging.info("SLAM monitor started in live mode.")
            logging.warning("TODO: Subscribe to /rtabmap/localization_pose and /rtabmap/info.")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None

    def update_from_pose(self, pose: Pose2D, quality_ok: bool = True) -> None:
        self.latest_pose = pose
        self._update_confidence(pose.covariance_norm, quality_ok)

    async def force_exploration(self, reason: str) -> None:
        if self.mode == SlamMode.EXPLORATION:
            return
        logging.warning("Forcing exploration mode: %s", reason)
        self.mode = SlamMode.EXPLORATION
        self._above_high_since = None
        await self._emit_mode_event()

    async def _mock_loop(self) -> None:
        start = time.time()
        while self._running:
            elapsed = (time.time() - start) % max(10.0, self.mock_cycle_s)
            cov_norm = self._mock_covariance(elapsed)
            quality_ok = elapsed < (self.mock_cycle_s * 0.85)

            # Simulated robot drift path for world-state summary.
            angle = elapsed / 6.0
            self.latest_pose = Pose2D(
                x=1.0 + 0.5 * math.cos(angle),
                y=0.5 + 0.5 * math.sin(angle),
                yaw=0.1 * math.sin(angle / 2.0),
                frame_id="map",
                covariance_norm=cov_norm,
            )
            self._update_confidence(cov_norm, quality_ok)
            await asyncio.sleep(0.5)

    def _mock_covariance(self, elapsed_s: float) -> float:
        high_until = self.mock_cycle_s * 0.30
        low_until = self.mock_cycle_s * 0.60
        if elapsed_s < high_until:
            return 0.08
        if elapsed_s < low_until:
            return 2.0
        return 0.15

    def _update_confidence(self, covariance_norm: float, quality_ok: bool) -> None:
        c_loc = max(0.0, min(1.0, 1.0 - (covariance_norm / max(1e-3, self.covariance_scale))))
        if not quality_ok:
            c_loc *= 0.5
        self.c_loc = c_loc
        self._evaluate_mode()

    def _evaluate_mode(self) -> None:
        now = time.time()
        switched = False

        if self.mode == SlamMode.LOCALIZATION and self.c_loc < self.t_low:
            self.mode = SlamMode.EXPLORATION
            self._above_high_since = None
            switched = True
            logging.warning("SLAM mode -> EXPLORATION (C_loc=%.2f < T_low=%.2f)", self.c_loc, self.t_low)

        elif self.mode == SlamMode.EXPLORATION:
            if self.c_loc > self.t_high:
                if self._above_high_since is None:
                    self._above_high_since = now
                elif now - self._above_high_since >= self.recover_hold_s:
                    self.mode = SlamMode.LOCALIZATION
                    self._above_high_since = None
                    switched = True
                    logging.info(
                        "SLAM mode -> LOCALIZATION (C_loc=%.2f held > T_high=%.2f for %.1fs)",
                        self.c_loc,
                        self.t_high,
                        self.recover_hold_s,
                    )
            else:
                self._above_high_since = None

        if switched:
            self._apply_rtabmap_mode()
            asyncio.create_task(self._emit_mode_event())

    async def _emit_mode_event(self) -> None:
        for callback in self._mode_callbacks:
            out = callback(self.mode, self.c_loc)
            if inspect.isawaitable(out):
                await out

    def _apply_rtabmap_mode(self) -> None:
        if self.mock_mode:
            return
        if self.mode == SlamMode.EXPLORATION:
            logging.warning(
                "TODO: Set RTAB-Map to mapping/exploration mode (e.g. ros2 param set or node restart)."
            )
        else:
            logging.warning(
                "TODO: Set RTAB-Map to localization-only mode (map update minimized)."
            )
