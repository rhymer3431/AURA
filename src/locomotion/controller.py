"""Robot and ONNX policy control logic."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from .constants import (
    ACTION_SCALE,
    DAMPING_PATTERNS,
    DEFAULT_JOINT_POS_PATTERNS,
    HEIGHT_SCAN_OFFSET,
    HEIGHT_SCAN_POINTS,
    STIFFNESS_PATTERNS,
)


def _pattern_matches(name: str, pattern: str) -> bool:
    glob = pattern.replace(".*", "*")
    return fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(name, glob + "*")


def _ordered_pattern_values(dof_names: list[str], pattern_values: dict[str, float], default: float) -> np.ndarray:
    ordered = []
    for dof_name in dof_names:
        value = default
        for pattern, pattern_value in pattern_values.items():
            if _pattern_matches(dof_name, pattern):
                value = pattern_value
                break
        ordered.append(value)
    return np.asarray(ordered, dtype=np.float32)


@dataclass(frozen=True)
class RobotBaseState:
    position_w: np.ndarray
    quat_wxyz: np.ndarray
    lin_vel_w: np.ndarray
    ang_vel_w: np.ndarray


class G1OnnxPolicyController:
    """Standalone ONNX locomotion controller for the G1 robot."""

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        onnx_path: str,
        position: np.ndarray,
        providers: list[str],
        decimation: int,
    ) -> None:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import define_prim, get_prim_at_path

        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            prim.GetReferences().AddReference(usd_path)

        self.robot = SingleArticulation(prim_path=prim_path, name="g1", position=position)
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.decimation = max(1, int(decimation))

        self.default_pos = np.zeros(0, dtype=np.float32)
        self.default_vel = np.zeros(0, dtype=np.float32)
        self.stiffness = np.zeros(0, dtype=np.float32)
        self.damping = np.zeros(0, dtype=np.float32)
        self.max_effort = np.zeros(0, dtype=np.float32)
        self.max_velocity = np.zeros(0, dtype=np.float32)

        self.action = np.zeros(0, dtype=np.float32)
        self.previous_action = np.zeros(0, dtype=np.float32)
        self.policy_counter = 0

    def initialize(self):
        from omni.physx import get_physx_simulation_interface

        self.robot.initialize()
        controller = self.robot.get_articulation_controller()
        controller.set_effort_modes("force")
        get_physx_simulation_interface().flush_changes()
        controller.switch_control_mode("position")

        dof_names = list(self.robot.dof_names)
        self.default_pos = _ordered_pattern_values(dof_names, DEFAULT_JOINT_POS_PATTERNS, default=0.0)
        self.default_vel = np.zeros(len(dof_names), dtype=np.float32)
        self.stiffness = _ordered_pattern_values(dof_names, STIFFNESS_PATTERNS, default=40.0)
        self.damping = _ordered_pattern_values(dof_names, DAMPING_PATTERNS, default=10.0)
        self.max_effort = np.full(len(dof_names), 300.0, dtype=np.float32)
        self.max_velocity = np.full(len(dof_names), 100.0, dtype=np.float32)

        self.robot._articulation_view.set_gains(self.stiffness.tolist(), self.damping.tolist())
        self.robot._articulation_view.set_max_efforts(self.max_effort.tolist())
        get_physx_simulation_interface().flush_changes()
        self.robot._articulation_view.set_max_joint_velocities(self.max_velocity.tolist())
        get_physx_simulation_interface().flush_changes()

        if hasattr(self.robot, "set_solver_position_iteration_count"):
            self.robot.set_solver_position_iteration_count(4)
        if hasattr(self.robot, "set_solver_velocity_iteration_count"):
            self.robot.set_solver_velocity_iteration_count(4)
        if hasattr(self.robot, "set_enabled_self_collisions"):
            self.robot.set_enabled_self_collisions(False)

        self.action = np.zeros(len(dof_names), dtype=np.float32)
        self.previous_action = np.zeros(len(dof_names), dtype=np.float32)
        self.policy_counter = 0
        self._validate_io()

    def reset(self):
        self.robot.post_reset()
        self.action.fill(0.0)
        self.previous_action.fill(0.0)
        self.policy_counter = 0

    def _validate_io(self):
        obs_dim = 12 + 3 * len(self.default_pos) + HEIGHT_SCAN_POINTS
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape

        if len(input_shape) >= 2 and isinstance(input_shape[-1], int) and int(input_shape[-1]) != obs_dim:
            raise RuntimeError(
                f"ONNX input obs dim mismatch: model expects {int(input_shape[-1])}, controller builds {obs_dim}."
            )
        if len(output_shape) >= 2 and isinstance(output_shape[-1], int) and int(output_shape[-1]) != len(
            self.default_pos
        ):
            raise RuntimeError(
                "ONNX output action dim mismatch: "
                f"model outputs {int(output_shape[-1])}, robot has {len(self.default_pos)} DOFs."
            )

    def get_base_state(self) -> RobotBaseState:
        lin_vel_w = np.asarray(self.robot.get_linear_velocity(), dtype=np.float32)
        ang_vel_w = np.asarray(self.robot.get_angular_velocity(), dtype=np.float32)
        pos_w, quat_w = self.robot.get_world_pose()
        return RobotBaseState(
            position_w=np.asarray(pos_w, dtype=np.float32).copy(),
            quat_wxyz=np.asarray(quat_w, dtype=np.float32).copy(),
            lin_vel_w=lin_vel_w.copy(),
            ang_vel_w=ang_vel_w.copy(),
        )

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        from isaacsim.core.utils.rotations import quat_to_rot_matrix

        base_state = self.get_base_state()
        lin_vel_w = base_state.lin_vel_w
        ang_vel_w = base_state.ang_vel_w
        pos_w = base_state.position_w
        quat_w = base_state.quat_wxyz

        rot_wb = quat_to_rot_matrix(quat_w)
        rot_bw = rot_wb.transpose()
        lin_vel_b = rot_bw @ lin_vel_w
        ang_vel_b = rot_bw @ ang_vel_w
        gravity_b = rot_bw @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)

        joint_pos = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
        joint_vel = np.asarray(self.robot.get_joint_velocities(), dtype=np.float32)
        height_value = np.clip(float(pos_w[2]) - HEIGHT_SCAN_OFFSET, -1.0, 1.0)
        height_scan = np.full(HEIGHT_SCAN_POINTS, height_value, dtype=np.float32)

        return np.concatenate(
            (
                lin_vel_b,
                ang_vel_b,
                gravity_b,
                command,
                joint_pos - self.default_pos,
                joint_vel - self.default_vel,
                self.previous_action,
                height_scan,
            ),
            dtype=np.float32,
        )

    def forward(self, step_index: int, command: np.ndarray):
        from isaacsim.core.utils.types import ArticulationAction

        if step_index % self.decimation == 0:
            obs = self._compute_observation(command)
            outputs = self.session.run([self.output_name], {self.input_name: obs.reshape(1, -1)})[0]
            self.action = np.asarray(outputs, dtype=np.float32).reshape(-1)
            self.previous_action = self.action.copy()

        target_positions = self.default_pos + (self.action * ACTION_SCALE)
        self.robot.apply_action(ArticulationAction(joint_positions=target_positions))
