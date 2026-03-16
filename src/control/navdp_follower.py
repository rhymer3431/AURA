from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from common.geometry import quat_wxyz_to_rot_matrix
from locomotion.paths import repo_dir, resolve_default_follower_policy_path, select_onnx_providers
from locomotion.policy_session import create_policy_session, infer_policy_backend


@dataclass(frozen=True)
class NavDPFollowerConfig:
    policy_path: str = ""
    onnx_device: str = "auto"
    max_vx: float = 0.5
    max_vy: float = 0.3
    max_wz: float = 0.8


@dataclass(frozen=True)
class NavDPFollowerResult:
    command: np.ndarray
    observation: np.ndarray


class NavDPFollower:
    def __init__(self, config: NavDPFollowerConfig, *, policy_session=None) -> None:
        self.config = config
        self.policy_path = os.path.abspath(
            config.policy_path or resolve_default_follower_policy_path(repo_dir())
        )
        if not os.path.isfile(self.policy_path):
            raise FileNotFoundError(f"NavDP follower policy not found: {self.policy_path}")
        backend = infer_policy_backend(self.policy_path)
        if backend != "onnxruntime":
            raise RuntimeError(f"NavDP follower policy must be ONNX, got: {self.policy_path}")
        providers = select_onnx_providers(str(config.onnx_device).strip().lower() or "auto")
        self._policy_session = policy_session or create_policy_session(
            self.policy_path,
            providers=providers,
            device_preference=str(config.onnx_device).strip().lower() or "auto",
        )
        self._validate_io()

    def close(self) -> None:
        self._policy_session.close()

    def compute_command(
        self,
        *,
        pose_command_b: np.ndarray,
        base_lin_vel_w: np.ndarray,
        base_ang_vel_w: np.ndarray,
        robot_quat_wxyz: np.ndarray,
    ) -> NavDPFollowerResult:
        observation = self._build_observation(
            pose_command_b=np.asarray(pose_command_b, dtype=np.float32),
            base_lin_vel_w=np.asarray(base_lin_vel_w, dtype=np.float32),
            base_ang_vel_w=np.asarray(base_ang_vel_w, dtype=np.float32),
            robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
        )
        outputs = np.asarray(self._policy_session.run(observation), dtype=np.float32).reshape(-1)
        command = np.asarray(
            [
                np.clip(float(outputs[0]), -float(self.config.max_vx), float(self.config.max_vx)),
                np.clip(float(outputs[1]), -float(self.config.max_vy), float(self.config.max_vy)),
                np.clip(float(outputs[2]), -float(self.config.max_wz), float(self.config.max_wz)),
            ],
            dtype=np.float32,
        )
        return NavDPFollowerResult(command=command, observation=observation.copy())

    def _build_observation(
        self,
        *,
        pose_command_b: np.ndarray,
        base_lin_vel_w: np.ndarray,
        base_ang_vel_w: np.ndarray,
        robot_quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        pose = np.asarray(pose_command_b, dtype=np.float32).reshape(-1)
        if pose.shape[0] != 4:
            raise ValueError(f"pose_command_b must have 4 elements, got shape={pose.shape}")
        rot_wb = quat_wxyz_to_rot_matrix(robot_quat_wxyz).astype(np.float32)
        rot_bw = rot_wb.transpose()
        lin_vel_b = rot_bw @ np.asarray(base_lin_vel_w, dtype=np.float32).reshape(3)
        ang_vel_b = rot_bw @ np.asarray(base_ang_vel_w, dtype=np.float32).reshape(3)
        gravity_b = rot_bw @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        return np.concatenate((lin_vel_b, ang_vel_b, gravity_b, pose), dtype=np.float32)

    def _validate_io(self) -> None:
        if tuple(self._policy_session.input_shape) != (1, 13):
            raise RuntimeError(
                f"NavDP follower input shape mismatch: expected (1, 13), got {self._policy_session.input_shape}."
            )
        if tuple(self._policy_session.output_shape) != (1, 3):
            raise RuntimeError(
                f"NavDP follower output shape mismatch: expected (1, 3), got {self._policy_session.output_shape}."
            )
