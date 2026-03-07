from __future__ import annotations

import math

import numpy as np


def quat_wxyz_to_rot_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if q.shape[0] < 4:
        raise ValueError(f"quat must have 4 elements, got shape={q.shape}")
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if q.shape[0] < 4:
        return 0.0
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def xy_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    a = np.asarray(point_a, dtype=np.float32).reshape(-1)
    b = np.asarray(point_b, dtype=np.float32).reshape(-1)
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError(f"xy_distance expects 2D points, got shapes {a.shape} and {b.shape}")
    return float(np.linalg.norm(a[:2] - b[:2]))


def within_xy_radius(point_a: np.ndarray, point_b: np.ndarray, radius_m: float) -> bool:
    if float(radius_m) < 0.0:
        raise ValueError(f"radius_m must be non-negative, got {radius_m}")
    return xy_distance(point_a, point_b) <= float(radius_m) + 1.0e-6


def world_goal_to_robot_frame(goal_xy: np.ndarray, robot_xy: np.ndarray, robot_yaw: float) -> np.ndarray:
    delta = np.asarray(goal_xy[:2], dtype=np.float32) - np.asarray(robot_xy[:2], dtype=np.float32)
    c = float(np.cos(robot_yaw))
    s = float(np.sin(robot_yaw))
    x_b = c * float(delta[0]) + s * float(delta[1])
    y_b = -s * float(delta[0]) + c * float(delta[1])
    return np.asarray([x_b, y_b], dtype=np.float32)


def trajectory_robot_to_world(
    trajectory_robot: np.ndarray,
    robot_pos_w: np.ndarray,
    robot_yaw: float,
) -> np.ndarray:
    traj = np.asarray(trajectory_robot, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"trajectory_robot must be [N,2+], got shape={traj.shape}")
    c = float(np.cos(robot_yaw))
    s = float(np.sin(robot_yaw))
    x0 = float(robot_pos_w[0])
    y0 = float(robot_pos_w[1])
    z0 = float(robot_pos_w[2]) if np.asarray(robot_pos_w).shape[0] >= 3 else 0.0
    world = np.zeros((traj.shape[0], 3), dtype=np.float32)
    for i, local in enumerate(traj):
        lx = float(local[0])
        ly = float(local[1])
        world[i, 0] = x0 + c * lx - s * ly
        world[i, 1] = y0 + s * lx + c * ly
        world[i, 2] = z0
    return world


def camera_axes_world(quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = quat_wxyz_to_rot_matrix(quat_wxyz)
    forward_w = -rot[:, 2]
    left_w = -rot[:, 0]
    up_w = rot[:, 1]
    return forward_w.astype(np.float32), left_w.astype(np.float32), up_w.astype(np.float32)


def world_goal_to_camera_pointgoal(
    goal_xyz_world: np.ndarray,
    camera_pos_world: np.ndarray,
    camera_quat_wxyz: np.ndarray,
) -> np.ndarray:
    goal = np.asarray(goal_xyz_world, dtype=np.float32).reshape(-1)
    cam_pos = np.asarray(camera_pos_world, dtype=np.float32).reshape(-1)
    forward_w, left_w, _ = camera_axes_world(camera_quat_wxyz)
    delta = goal[:3] - cam_pos[:3]
    goal_x = float(np.dot(delta, forward_w))
    goal_y = float(np.dot(delta, left_w))
    return np.asarray([goal_x, goal_y], dtype=np.float32)


def trajectory_camera_to_world(
    trajectory_local: np.ndarray,
    camera_pos_world: np.ndarray,
    camera_quat_wxyz: np.ndarray,
    use_trajectory_z: bool,
) -> np.ndarray:
    traj = np.asarray(trajectory_local, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"trajectory_local must be [N,2+], got shape={traj.shape}")
    cam_pos = np.asarray(camera_pos_world, dtype=np.float32).reshape(-1)
    forward_w, left_w, up_w = camera_axes_world(camera_quat_wxyz)
    world = np.zeros((traj.shape[0], 3), dtype=np.float32)
    for i in range(traj.shape[0]):
        x = float(traj[i, 0])
        y = float(traj[i, 1])
        z = float(traj[i, 2]) if use_trajectory_z and traj.shape[1] >= 3 else 0.0
        world[i, :] = cam_pos[:3] + x * forward_w + y * left_w + z * up_w
    return world


def world_goal_to_local_pointgoal(
    goal_xyz_world: np.ndarray,
    *,
    pointgoal_frame: str,
    camera_pos_world: np.ndarray | None = None,
    camera_quat_wxyz: np.ndarray | None = None,
    robot_pos_world: np.ndarray | None = None,
    robot_yaw: float | None = None,
) -> np.ndarray:
    frame = str(pointgoal_frame).strip().lower()
    if frame == "camera":
        if camera_pos_world is None or camera_quat_wxyz is None:
            raise ValueError("camera frame requires camera_pos_world and camera_quat_wxyz")
        return world_goal_to_camera_pointgoal(
            goal_xyz_world=goal_xyz_world,
            camera_pos_world=camera_pos_world,
            camera_quat_wxyz=camera_quat_wxyz,
        )
    if frame == "robot":
        if robot_pos_world is None or robot_yaw is None:
            raise ValueError("robot frame requires robot_pos_world and robot_yaw")
        goal_xy = np.asarray(goal_xyz_world, dtype=np.float32).reshape(-1)[:2]
        robot_xy = np.asarray(robot_pos_world, dtype=np.float32).reshape(-1)[:2]
        return world_goal_to_robot_frame(goal_xy=goal_xy, robot_xy=robot_xy, robot_yaw=float(robot_yaw))
    raise ValueError(f"unsupported pointgoal_frame: {pointgoal_frame}")


def trajectory_local_to_world(
    trajectory_local: np.ndarray,
    *,
    pointgoal_frame: str,
    use_trajectory_z: bool,
    camera_pos_world: np.ndarray | None = None,
    camera_quat_wxyz: np.ndarray | None = None,
    robot_pos_world: np.ndarray | None = None,
    robot_yaw: float | None = None,
) -> np.ndarray:
    frame = str(pointgoal_frame).strip().lower()
    if frame == "camera":
        if camera_pos_world is None or camera_quat_wxyz is None:
            raise ValueError("camera frame requires camera_pos_world and camera_quat_wxyz")
        return trajectory_camera_to_world(
            trajectory_local=trajectory_local,
            camera_pos_world=camera_pos_world,
            camera_quat_wxyz=camera_quat_wxyz,
            use_trajectory_z=use_trajectory_z,
        )
    if frame == "robot":
        if robot_pos_world is None or robot_yaw is None:
            raise ValueError("robot frame requires robot_pos_world and robot_yaw")
        return trajectory_robot_to_world(
            trajectory_robot=trajectory_local,
            robot_pos_w=np.asarray(robot_pos_world, dtype=np.float32).reshape(-1),
            robot_yaw=float(robot_yaw),
        )
    raise ValueError(f"unsupported pointgoal_frame: {pointgoal_frame}")


def normalize_navdp_trajectory(raw_trajectory: np.ndarray) -> np.ndarray:
    traj = np.asarray(raw_trajectory, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[0]
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"invalid trajectory shape: {traj.shape}")
    return traj


def step_pose_towards_target(
    current_pos_xyz: np.ndarray,
    current_yaw: float,
    target_xy: np.ndarray,
    max_step_m: float,
    max_yaw_step_rad: float,
) -> tuple[np.ndarray, float]:
    current = np.asarray(current_pos_xyz, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    delta = target[:2] - current[:2]
    dist = float(np.linalg.norm(delta))
    if dist < 1.0e-6:
        return current.copy(), float(current_yaw)

    direction = delta / dist
    move = min(float(max_step_m), dist)
    next_pos = current.copy()
    next_pos[0] += float(direction[0]) * move
    next_pos[1] += float(direction[1]) * move

    target_yaw = float(math.atan2(float(direction[1]), float(direction[0])))
    yaw_error = wrap_to_pi(target_yaw - float(current_yaw))
    yaw_step = float(np.clip(yaw_error, -float(max_yaw_step_rad), float(max_yaw_step_rad)))
    next_yaw = wrap_to_pi(float(current_yaw) + yaw_step)
    return next_pos, next_yaw
