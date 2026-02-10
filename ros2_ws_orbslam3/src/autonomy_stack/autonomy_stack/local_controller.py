from __future__ import annotations

import math
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist

from .params import ControllerParams


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class LocalController:
    def __init__(self, params: ControllerParams) -> None:
        self._p = params
        self._last_cmd = Twist()
        self._last_time: Optional[rclpy.time.Time] = None

    def reset(self) -> None:
        self._last_cmd = Twist()
        self._last_time = None

    def _limit_accel(self, target: Twist, now: rclpy.time.Time) -> Twist:
        if self._last_time is None:
            self._last_time = now
            self._last_cmd = target
            return target

        dt = max(1e-3, (now - self._last_time).nanoseconds * 1e-9)
        self._last_time = now

        out = Twist()
        max_dv = float(self._p.max_linear_accel) * dt
        max_dw = float(self._p.max_angular_accel) * dt
        dv = target.linear.x - self._last_cmd.linear.x
        dw = target.angular.z - self._last_cmd.angular.z
        out.linear.x = self._last_cmd.linear.x + _clamp(dv, -max_dv, max_dv)
        out.angular.z = self._last_cmd.angular.z + _clamp(dw, -max_dw, max_dw)
        self._last_cmd = out
        return out

    def compute_cmd(
        self,
        now: rclpy.time.Time,
        path_xy: List[Tuple[float, float]],
        collision: bool,
    ) -> Tuple[Twist, bool]:
        cmd = Twist()
        goal_reached = False

        if collision:
            return cmd, False

        if not path_xy:
            return cmd, False

        # robot is at (0,0) in base_link
        distances = [math.hypot(p[0], p[1]) for p in path_xy]
        goal_dist = distances[-1]
        if goal_dist <= float(self._p.goal_tolerance):
            goal_reached = True
            return cmd, goal_reached

        lookahead = float(self._p.lookahead_distance)
        target = path_xy[-1]
        for p, d in zip(path_xy, distances):
            if d >= lookahead:
                target = p
                break

        angle = _wrap(math.atan2(target[1], target[0]))
        linear = float(self._p.linear_speed)
        if abs(angle) > 0.7:
            linear *= 0.3
        if abs(angle) > 1.2:
            linear = 0.0

        angular = float(self._p.angular_kp) * angle
        linear = _clamp(linear, -self._p.max_linear_speed, self._p.max_linear_speed)
        angular = _clamp(angular, -self._p.max_angular_speed, self._p.max_angular_speed)

        cmd.linear.x = linear
        cmd.angular.z = angular
        cmd = self._limit_accel(cmd, now)
        return cmd, goal_reached
