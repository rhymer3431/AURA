from __future__ import annotations

from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist

from .params import RecoveryParams


class RecoveryManager:
    def __init__(self, params: RecoveryParams) -> None:
        self._p = params
        self._state = "IDLE"
        self._state_start: Optional[rclpy.time.Time] = None

    def reset(self) -> None:
        self._state = "IDLE"
        self._state_start = None

    def _elapsed(self, now: rclpy.time.Time) -> float:
        if self._state_start is None:
            return 0.0
        return (now - self._state_start).nanoseconds * 1e-9

    def _twist(self, linear: float, angular: float) -> Twist:
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        return cmd

    def update(
        self,
        now: rclpy.time.Time,
        tracking_ok: bool,
        stuck: bool,
    ) -> Tuple[bool, Twist]:
        if not tracking_ok:
            if self._state != "SLAM_LOST":
                self._state = "SLAM_LOST"
                self._state_start = now
            if self._elapsed(now) <= float(self._p.slam_lost_rotate_time):
                return True, self._twist(0.0, self._p.slam_lost_rotate_speed)
            self._state = "IDLE"
            self._state_start = None
            return False, Twist()

        if self._state == "SLAM_LOST":
            self._state = "IDLE"
            self._state_start = None

        if self._state == "IDLE" and stuck:
            self._state = "STUCK_BACKUP"
            self._state_start = now

        if self._state == "STUCK_BACKUP":
            if self._elapsed(now) <= float(self._p.backup_time):
                return True, self._twist(self._p.backup_speed, 0.0)
            self._state = "STUCK_ROTATE"
            self._state_start = now

        if self._state == "STUCK_ROTATE":
            if self._elapsed(now) <= float(self._p.rotate_time):
                return True, self._twist(0.0, self._p.rotate_speed)
            self._state = "IDLE"
            self._state_start = None

        return False, Twist()
