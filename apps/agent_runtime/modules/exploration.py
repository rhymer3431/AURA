from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from .contracts import Pose2D, SlamMode
from .nav2_client import Nav2Client
from .slam_monitor import SLAMMonitor


class ExplorationBehavior:
    """Simple exploration fallback: waypoint sweep until relocalization."""

    def __init__(self, cfg: Dict) -> None:
        self.radius_m = float(cfg.get("radius_m", 1.2))
        self.max_cycles = int(cfg.get("max_cycles", 3))
        self.waypoint_timeout_s = float(cfg.get("waypoint_timeout_s", 12.0))

    def _waypoints(self, center: Optional[Pose2D]) -> List[Pose2D]:
        base = center or Pose2D(0.0, 0.0, 0.0)
        r = self.radius_m
        return [
            Pose2D(base.x + r, base.y, 0.0),
            Pose2D(base.x, base.y + r, 1.57),
            Pose2D(base.x - r, base.y, 3.14),
            Pose2D(base.x, base.y - r, -1.57),
        ]

    async def recover(self, nav_client: Nav2Client, slam_monitor: SLAMMonitor) -> bool:
        logging.warning("Exploration mode started. Running waypoint sweep for relocalization.")
        for cycle in range(self.max_cycles):
            if slam_monitor.mode != SlamMode.EXPLORATION:
                return True
            for wp in self._waypoints(slam_monitor.latest_pose):
                if slam_monitor.mode != SlamMode.EXPLORATION:
                    return True
                result = await nav_client.navigate_to_pose(
                    wp, timeout_s=self.waypoint_timeout_s, allow_low_confidence=True
                )
                if result.success:
                    logging.info("Exploration waypoint reached.")
                else:
                    logging.warning("Exploration waypoint failed: %s", result.reason)
                await asyncio.sleep(0.2)
            logging.info("Exploration cycle %s/%s complete.", cycle + 1, self.max_cycles)
        return slam_monitor.mode != SlamMode.EXPLORATION
