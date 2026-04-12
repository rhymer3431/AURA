"""Quaternion and rotation helpers shared within perception camera control."""

from __future__ import annotations

import numpy as np


def quaternion_from_axis_angle_wxyz(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_vec = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_vec))
    if norm <= 1e-12:
        raise ValueError("axis must be non-zero")
    unit_axis = axis_vec / norm
    half_angle = 0.5 * float(angle_rad)
    sin_half = np.sin(half_angle)
    return np.asarray(
        (
            np.cos(half_angle),
            unit_axis[0] * sin_half,
            unit_axis[1] * sin_half,
            unit_axis[2] * sin_half,
        ),
        dtype=np.float64,
    )


def quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(lhs, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(rhs, dtype=np.float64).reshape(4)
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )


def rotation_matrix_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.asarray(
        (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        ),
        dtype=np.float64,
    )
