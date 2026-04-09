"""Geometry helpers for NavDP plan following."""

from __future__ import annotations

import math

import numpy as np


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""

    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def normalize_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return a normalized scalar-first quaternion."""

    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero.")
    return q / norm


def quaternion_multiply_wxyz(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    """Compose two scalar-first quaternions as R(lhs) @ R(rhs)."""

    w1, x1, y1, z1 = normalize_quaternion_wxyz(lhs_wxyz)
    w2, x2, y2, z2 = normalize_quaternion_wxyz(rhs_wxyz)
    return normalize_quaternion_wxyz(
        np.asarray(
            (
                (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2),
                (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2),
                (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2),
                (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2),
            ),
            dtype=np.float64,
        )
    )


def quaternion_from_axis_angle_wxyz(axis_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    """Build a scalar-first quaternion from a 3D axis and rotation angle."""

    axis = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("Axis norm must be non-zero.")
    axis = axis / norm
    half_angle = 0.5 * float(angle_rad)
    sin_half = math.sin(half_angle)
    return normalize_quaternion_wxyz(
        np.asarray((math.cos(half_angle), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half), dtype=np.float64)
    )


def rotation_matrix_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert a scalar-first quaternion into a 3x3 rotation matrix."""

    w, x, y, z = normalize_quaternion_wxyz(quaternion_wxyz)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def yaw_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> float:
    """Extract the world yaw angle from a scalar-first quaternion."""

    rot = rotation_matrix_from_quaternion_wxyz(quaternion_wxyz)
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def camera_plan_to_world_xy(
    trajectory_camera: np.ndarray,
    camera_pos_w: np.ndarray,
    camera_rot_w: np.ndarray,
) -> np.ndarray:
    """Transform NavDP camera-frame trajectory points into world XY points."""

    plan = np.asarray(trajectory_camera, dtype=np.float64)
    if plan.ndim != 2 or plan.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    points_world = []
    camera_pos_w = np.asarray(camera_pos_w, dtype=np.float64).reshape(3)
    camera_rot_w = np.asarray(camera_rot_w, dtype=np.float64).reshape(3, 3)
    for point in plan:
        local = np.asarray((float(point[0]), float(point[1]), 0.0), dtype=np.float64)
        world = camera_pos_w + (camera_rot_w @ local)
        points_world.append(world[:2])
    return np.asarray(points_world, dtype=np.float32)


def world_xy_to_body_xy(path_world_xy: np.ndarray, base_pos_w: np.ndarray, base_yaw: float) -> np.ndarray:
    """Transform a world-frame XY path into the robot body frame."""

    path_world_xy = np.asarray(path_world_xy, dtype=np.float64)
    if path_world_xy.ndim != 2 or path_world_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    base_xy = np.asarray(base_pos_w, dtype=np.float64).reshape(3)[:2]
    delta_world = path_world_xy - base_xy[None, :]
    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    rot_bw = np.asarray(((cos_yaw, sin_yaw), (-sin_yaw, cos_yaw)), dtype=np.float64)
    return (delta_world @ rot_bw.T).astype(np.float32)


def point_goal_world_from_frame(
    goal_xy: np.ndarray,
    goal_frame: str,
    start_pos_w: np.ndarray,
    start_yaw: float,
) -> np.ndarray:
    """Resolve a point goal into world XY coordinates."""

    goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    if goal_frame == "world":
        return goal_xy.astype(np.float32)
    if goal_frame != "start":
        raise ValueError(f"Unsupported point goal frame: {goal_frame}")

    base_xy = np.asarray(start_pos_w, dtype=np.float64).reshape(3)[:2]
    cos_yaw = math.cos(start_yaw)
    sin_yaw = math.sin(start_yaw)
    rot_wb = np.asarray(((cos_yaw, -sin_yaw), (sin_yaw, cos_yaw)), dtype=np.float64)
    target_world = base_xy + (rot_wb @ goal_xy)
    return target_world.astype(np.float32)


def point_goal_body_from_world(goal_world_xy: np.ndarray, base_pos_w: np.ndarray, base_yaw: float) -> np.ndarray:
    """Convert a world-frame target point into the robot body frame."""

    return world_xy_to_body_xy(
        np.asarray(goal_world_xy, dtype=np.float32).reshape(1, 2),
        base_pos_w=np.asarray(base_pos_w, dtype=np.float32),
        base_yaw=base_yaw,
    )[0]
