from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable, Dict, Optional

from .contracts import NavResult, Pose2D


class Nav2Client:
    """Nav2 NavigateToPose adapter with mock fallback."""

    def __init__(self, cfg: Dict, get_confidence: Optional[Callable[[], float]] = None) -> None:
        self.mock_mode = bool(cfg.get("mock_mode", True))
        self.default_timeout_s = float(cfg.get("default_timeout_s", 20.0))
        self.mock_failure_rate = float(cfg.get("mock_failure_rate", 0.0))
        self.pose_uncertain_threshold = float(cfg.get("pose_uncertain_threshold", 0.35))
        self.get_confidence = get_confidence

    async def navigate_to_pose(
        self,
        goal_pose: Pose2D,
        timeout_s: Optional[float] = None,
        pause_event: Optional[asyncio.Event] = None,
        allow_low_confidence: bool = False,
    ) -> NavResult:
        if self.mock_mode:
            return await self._mock_navigate(
                goal_pose,
                timeout_s or self.default_timeout_s,
                pause_event,
                allow_low_confidence,
            )

        logging.warning("TODO: Connect to nav2_msgs/action/NavigateToPose action server.")
        # TODO: Implement real action client using rclpy action.
        await asyncio.sleep(0.2)
        return NavResult(success=False, reason="NOT_IMPLEMENTED")

    async def _mock_navigate(
        self,
        goal_pose: Pose2D,
        timeout_s: float,
        pause_event: Optional[asyncio.Event],
        allow_low_confidence: bool,
    ) -> NavResult:
        c_loc = self.get_confidence() if self.get_confidence is not None else 1.0
        if c_loc < self.pose_uncertain_threshold and not allow_low_confidence:
            return NavResult(False, "POSE_UNCERTAIN")

        travel_s = 1.2 + random.random() * 2.0
        deadline = time.time() + timeout_s
        logging.info(
            "Nav2 mock start: x=%.2f y=%.2f yaw=%.2f timeout=%.1fs",
            goal_pose.x,
            goal_pose.y,
            goal_pose.yaw,
            timeout_s,
        )

        while time.time() < deadline and travel_s > 0:
            if pause_event is not None and not pause_event.is_set():
                logging.info("Nav2 paused due to SLAM exploration mode.")
                await pause_event.wait()
            await asyncio.sleep(0.2)
            travel_s -= 0.2

        if time.time() >= deadline:
            return NavResult(False, "TIMEOUT")
        if random.random() < self.mock_failure_rate:
            return NavResult(False, random.choice(["NO_PATH", "TIMEOUT"]))
        return NavResult(True, "OK")
