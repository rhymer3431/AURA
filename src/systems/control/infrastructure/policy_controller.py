"""Robot and policy control logic."""

from __future__ import annotations

import fnmatch
import math

import carb
import numpy as np
from isaacsim.core.prims import RigidPrim, SingleArticulation
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from omni.physx import get_physx_scene_query_interface, get_physx_simulation_interface

from systems.control.domain.constants import (
    DEFAULT_JOINT_POS_PATTERNS,
    DEFAULT_PHYSICS_DT,
    FEET_CONTACT_HISTORY_LENGTH,
    FEET_CONTACT_THRESHOLD,
    HEIGHT_SCAN_OFFSET,
    HEIGHT_SCAN_RAYCAST_DISTANCE,
    HEIGHT_SCAN_RAY_START_Z,
    LEGACY_ACTION_SCALE,
    LEGACY_DAMPING_PATTERNS,
    LEGACY_HEIGHT_SCAN_RESOLUTION,
    LEGACY_HEIGHT_SCAN_SIZE,
    LEGACY_STIFFNESS_PATTERNS,
    TUNED_ACTION_SCALE,
    TUNED_DAMPING_PATTERNS,
    TUNED_HEIGHT_SCAN_RESOLUTION,
    TUNED_HEIGHT_SCAN_SIZE,
    TUNED_STIFFNESS_PATTERNS,
)
from systems.control.infrastructure.policy_session import create_policy_session, infer_policy_backend
from systems.world_state.api.observation_layout import PolicyObservationLayout, infer_policy_observation_layout


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


def _height_scan_point_count(size: tuple[float, float], resolution: float) -> int:
    x_count = int(round(float(size[0]) / float(resolution))) + 1
    y_count = int(round(float(size[1]) / float(resolution))) + 1
    return x_count * y_count


def _build_height_scan_grid(size: tuple[float, float], resolution: float) -> np.ndarray:
    x_count = int(round(float(size[0]) / float(resolution))) + 1
    y_count = int(round(float(size[1]) / float(resolution))) + 1
    x_coords = np.linspace(-float(size[0]) * 0.5, float(size[0]) * 0.5, x_count, dtype=np.float32)
    y_coords = np.linspace(-float(size[1]) * 0.5, float(size[1]) * 0.5, y_count, dtype=np.float32)
    return np.asarray([(x, y) for x in x_coords for y in y_coords], dtype=np.float32)


def _select_runtime_preset(expected_height_scan_points: int | None):
    legacy_count = _height_scan_point_count(LEGACY_HEIGHT_SCAN_SIZE, LEGACY_HEIGHT_SCAN_RESOLUTION)
    tuned_count = _height_scan_point_count(TUNED_HEIGHT_SCAN_SIZE, TUNED_HEIGHT_SCAN_RESOLUTION)
    if expected_height_scan_points == 0:
        return {
            "name": "flat",
            "action_scale": LEGACY_ACTION_SCALE,
            "height_scan_size": None,
            "height_scan_resolution": None,
            "stiffness_patterns": LEGACY_STIFFNESS_PATTERNS,
            "damping_patterns": LEGACY_DAMPING_PATTERNS,
            "solver_position_iterations": 8,
            "solver_velocity_iterations": 4,
        }
    if expected_height_scan_points == legacy_count:
        return {
            "name": "legacy",
            "action_scale": LEGACY_ACTION_SCALE,
            "height_scan_size": LEGACY_HEIGHT_SCAN_SIZE,
            "height_scan_resolution": LEGACY_HEIGHT_SCAN_RESOLUTION,
            "stiffness_patterns": LEGACY_STIFFNESS_PATTERNS,
            "damping_patterns": LEGACY_DAMPING_PATTERNS,
            "solver_position_iterations": 4,
            "solver_velocity_iterations": 4,
        }
    if expected_height_scan_points == tuned_count or expected_height_scan_points is None:
        return {
            "name": "tuned",
            "action_scale": TUNED_ACTION_SCALE,
            "height_scan_size": TUNED_HEIGHT_SCAN_SIZE,
            "height_scan_resolution": TUNED_HEIGHT_SCAN_RESOLUTION,
            "stiffness_patterns": TUNED_STIFFNESS_PATTERNS,
            "damping_patterns": TUNED_DAMPING_PATTERNS,
            "solver_position_iterations": 10,
            "solver_velocity_iterations": 8,
        }
    return None


class G1PolicyController:
    """Standalone locomotion controller for the G1 robot."""

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        policy_path: str,
        position: np.ndarray,
        decimation: int,
        physics_dt: float | None = None,
        providers: list[str] | None = None,
        device_preference: str = "auto",
        action_scale: float | None = None,
        height_scan_size: tuple[float, float] | None = None,
        height_scan_resolution: float | None = None,
        height_scan_offset: float | None = None,
        default_joint_pos_patterns: dict[str, float] | None = None,
        stiffness_patterns: dict[str, float] | None = None,
        damping_patterns: dict[str, float] | None = None,
        solver_position_iterations: int | None = None,
        solver_velocity_iterations: int | None = None,
        height_scan_enabled: bool | None = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            prim.GetReferences().AddReference(usd_path)

        self.robot = SingleArticulation(prim_path=prim_path, name="g1", position=position)
        self.policy_path = policy_path
        self.session = create_policy_session(
            policy_path=policy_path,
            providers=list(providers or []),
            device_preference=device_preference,
        )
        self.backend = infer_policy_backend(policy_path)
        self.input_name = self.session.input_name
        self.output_name = self.session.output_name
        self.decimation = max(1, int(decimation))
        self.physics_dt = float(physics_dt) if physics_dt is not None else DEFAULT_PHYSICS_DT
        self.robot_prim_path = prim_path.rstrip("/")
        self.scene_query = get_physx_scene_query_interface()

        self.default_pos = np.zeros(0, dtype=np.float32)
        self.default_vel = np.zeros(0, dtype=np.float32)
        self.stiffness = np.zeros(0, dtype=np.float32)
        self.damping = np.zeros(0, dtype=np.float32)
        self.max_effort = np.zeros(0, dtype=np.float32)
        self.max_velocity = np.zeros(0, dtype=np.float32)

        self.action = np.zeros(0, dtype=np.float32)
        self.previous_action = np.zeros(0, dtype=np.float32)
        self.policy_counter = 0
        self.action_scale = float(action_scale) if action_scale is not None else None
        self.height_scan_size = tuple(height_scan_size) if height_scan_size is not None else None
        self.height_scan_resolution = float(height_scan_resolution) if height_scan_resolution is not None else None
        self.height_scan_offset = HEIGHT_SCAN_OFFSET if height_scan_offset is None else float(height_scan_offset)
        self.default_joint_pos_patterns = dict(default_joint_pos_patterns or DEFAULT_JOINT_POS_PATTERNS)
        self.stiffness_patterns_override = dict(stiffness_patterns) if stiffness_patterns is not None else None
        self.damping_patterns_override = dict(damping_patterns) if damping_patterns is not None else None
        self.solver_position_iterations_override = (
            int(solver_position_iterations) if solver_position_iterations is not None else None
        )
        self.solver_velocity_iterations_override = (
            int(solver_velocity_iterations) if solver_velocity_iterations is not None else None
        )
        self.height_scan_enabled = height_scan_enabled
        self.height_scan_grid = np.zeros((0, 2), dtype=np.float32)
        self.feet_contact_view = None
        self.feet_contact_body_names: list[str] = []
        self.feet_contact_threshold = FEET_CONTACT_THRESHOLD
        self.feet_contact_history = np.zeros((0, 0, 3), dtype=np.float32)
        self.feet_contact_history_index = 0
        self.observation_layout: PolicyObservationLayout | None = None
        self.solver_position_iterations = 10
        self.solver_velocity_iterations = 8

    def _expected_input_obs_dim(self) -> int | None:
        input_shape = self.session.input_shape
        if len(input_shape) >= 2 and isinstance(input_shape[-1], int):
            return int(input_shape[-1])
        return None

    def _expected_height_scan_points(self) -> int | None:
        return self._infer_observation_layout().height_scan_points

    def _expected_feet_contact_dim(self) -> int:
        return self._infer_observation_layout().feet_contact_dim

    def _infer_observation_layout(self) -> PolicyObservationLayout:
        if self.observation_layout is None:
            self.observation_layout = infer_policy_observation_layout(
                input_obs_dim=self._expected_input_obs_dim(),
                dof_count=len(self.default_pos),
            )
        return self.observation_layout

    def _configure_feet_contact_state(self) -> None:
        expected_feet_contact_dim = self._expected_feet_contact_dim()
        if expected_feet_contact_dim == 0:
            self.feet_contact_view = None
            self.feet_contact_body_names = []
            self.feet_contact_history = np.zeros((0, 0, 3), dtype=np.float32)
            self.feet_contact_history_index = 0
            return

        foot_body_names = [
            body_name
            for body_name in list(self.robot._articulation_view.body_names)
            if _pattern_matches(body_name, ".*_ankle_roll_link")
        ]
        if len(foot_body_names) != expected_feet_contact_dim:
            raise RuntimeError(
                "Policy expects feet_contact_state observation dims that do not match the robot links: "
                f"expected {expected_feet_contact_dim}, matched {foot_body_names}."
            )

        foot_body_paths = [f"{self.robot_prim_path}/{body_name}" for body_name in foot_body_names]
        missing_paths = [path for path in foot_body_paths if not get_prim_at_path(path).IsValid()]
        if missing_paths:
            raise RuntimeError(
                "Unable to create feet contact view because the expected foot link prims were not found: "
                f"{missing_paths}"
            )

        self.feet_contact_body_names = foot_body_names
        self.feet_contact_history = np.zeros(
            (FEET_CONTACT_HISTORY_LENGTH, len(self.feet_contact_body_names), 3),
            dtype=np.float32,
        )
        self.feet_contact_history_index = 0
        # Isaac Sim 5.1 requires one filter entry per tracked rigid body even when
        # only aggregate net contact forces are queried.
        contact_filter_prim_paths_expr = [[] for _ in foot_body_paths]
        try:
            self.feet_contact_view = RigidPrim(
                prim_paths_expr=foot_body_paths,
                name="g1_feet_contact_view",
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=contact_filter_prim_paths_expr,
            )
            self.feet_contact_view.initialize()
        except Exception as exc:
            self.feet_contact_view = None
            carb.log_warn(
                "Feet contact view initialization failed; continuing with zero-filled "
                f"feet_contact_state observations. Error: {exc}"
            )

    def _compute_feet_contact_state(self) -> np.ndarray:
        if self.feet_contact_view is None or len(self.feet_contact_body_names) == 0:
            return np.zeros(0, dtype=np.float32)

        contact_norm_history = np.linalg.norm(self.feet_contact_history, axis=-1)
        return (contact_norm_history.max(axis=0) > float(self.feet_contact_threshold)).astype(np.float32)

    def _update_feet_contact_history(self) -> None:
        if self.feet_contact_view is None or len(self.feet_contact_body_names) == 0:
            return

        contact_forces = self.feet_contact_view.get_net_contact_forces(dt=self.physics_dt)
        if hasattr(contact_forces, "detach"):
            contact_forces = contact_forces.detach()
        if hasattr(contact_forces, "cpu"):
            contact_forces = contact_forces.cpu()
        contact_forces = np.asarray(contact_forces, dtype=np.float32).reshape(-1, 3)
        if contact_forces.shape[0] != len(self.feet_contact_body_names):
            raise RuntimeError(
                "Feet contact view returned an unexpected number of bodies: "
                f"expected {len(self.feet_contact_body_names)}, got {contact_forces.shape[0]}."
            )

        self.feet_contact_history[self.feet_contact_history_index] = contact_forces
        self.feet_contact_history_index = (self.feet_contact_history_index + 1) % len(self.feet_contact_history)

    def _configure_runtime_parameters(self) -> tuple[dict[str, float], dict[str, float]]:
        expected_height_scan_points = self._expected_height_scan_points()
        preset = _select_runtime_preset(expected_height_scan_points)
        if preset is None:
            raise RuntimeError(
                "Unsupported policy engine observation shape for automatic height scan selection. "
                "Pass --height_scan_size and --height_scan_resolution explicitly."
            )

        if self.height_scan_size is None:
            self.height_scan_size = preset["height_scan_size"]
        if self.height_scan_resolution is None:
            self.height_scan_resolution = preset["height_scan_resolution"]
        if self.action_scale is None:
            self.action_scale = float(preset["action_scale"])

        height_scan_enabled = self.height_scan_enabled
        if height_scan_enabled is None:
            height_scan_enabled = self.height_scan_size is not None and self.height_scan_resolution is not None
            if expected_height_scan_points == 0 and self.height_scan_size is None and self.height_scan_resolution is None:
                height_scan_enabled = False

        if height_scan_enabled:
            if self.height_scan_size is None or self.height_scan_resolution is None:
                raise RuntimeError(
                    "Height scan is enabled but the scan size/resolution are undefined. "
                    "Pass --height_scan_size and --height_scan_resolution explicitly."
                )
            self.height_scan_grid = _build_height_scan_grid(self.height_scan_size, self.height_scan_resolution)
        else:
            self.height_scan_grid = np.zeros((0, 2), dtype=np.float32)

        if expected_height_scan_points is not None and len(self.height_scan_grid) != expected_height_scan_points:
            raise RuntimeError(
                "Height scan configuration does not match policy engine input shape: "
                f"expected {expected_height_scan_points} points, built {len(self.height_scan_grid)}."
            )

        self._configure_feet_contact_state()

        self.solver_position_iterations = (
            self.solver_position_iterations_override
            if self.solver_position_iterations_override is not None
            else int(preset["solver_position_iterations"])
        )
        self.solver_velocity_iterations = (
            self.solver_velocity_iterations_override
            if self.solver_velocity_iterations_override is not None
            else int(preset["solver_velocity_iterations"])
        )
        stiffness_patterns = self.stiffness_patterns_override or preset["stiffness_patterns"]
        damping_patterns = self.damping_patterns_override or preset["damping_patterns"]
        return stiffness_patterns, damping_patterns

    def initialize(self):
        self.robot.initialize()
        print(f"[INFO] G1 locomotion policy: {self.policy_path}")
        print(f"[INFO] G1 locomotion backend: {self.backend}")
        print(f"[INFO] G1 locomotion policy I/O: input={self.session.input_shape} output={self.session.output_shape}")
        controller = self.robot.get_articulation_controller()
        controller.set_effort_modes("force")
        get_physx_simulation_interface().flush_changes()
        controller.switch_control_mode("position")

        dof_names = list(self.robot.dof_names)
        self.default_pos = _ordered_pattern_values(dof_names, self.default_joint_pos_patterns, default=0.0)
        self.default_vel = np.zeros(len(dof_names), dtype=np.float32)
        stiffness_patterns, damping_patterns = self._configure_runtime_parameters()
        self.stiffness = _ordered_pattern_values(dof_names, stiffness_patterns, default=40.0)
        self.damping = _ordered_pattern_values(dof_names, damping_patterns, default=10.0)
        self.max_effort = np.full(len(dof_names), 300.0, dtype=np.float32)
        self.max_velocity = np.full(len(dof_names), 100.0, dtype=np.float32)

        self.robot._articulation_view.set_gains(self.stiffness.tolist(), self.damping.tolist())
        self.robot._articulation_view.set_max_efforts(self.max_effort.tolist())
        get_physx_simulation_interface().flush_changes()
        self.robot._articulation_view.set_max_joint_velocities(self.max_velocity.tolist())
        get_physx_simulation_interface().flush_changes()

        if hasattr(self.robot, "set_solver_position_iteration_count"):
            self.robot.set_solver_position_iteration_count(self.solver_position_iterations)
        if hasattr(self.robot, "set_solver_velocity_iteration_count"):
            self.robot.set_solver_velocity_iteration_count(self.solver_velocity_iterations)
        if hasattr(self.robot, "set_enabled_self_collisions"):
            self.robot.set_enabled_self_collisions(False)

        self.action = np.zeros(len(dof_names), dtype=np.float32)
        self.previous_action = np.zeros(len(dof_names), dtype=np.float32)
        self.policy_counter = 0
        if self.feet_contact_body_names:
            print(f"[INFO] Feet contact observation bodies: {self.feet_contact_body_names}")
        self._validate_io()

    def reset(self):
        self.robot.post_reset()
        self.action.fill(0.0)
        self.previous_action.fill(0.0)
        self.policy_counter = 0
        if self.feet_contact_history.size:
            self.feet_contact_history.fill(0.0)
            self.feet_contact_history_index = 0

    def _validate_io(self):
        obs_dim = 12 + 3 * len(self.default_pos) + len(self.height_scan_grid) + len(self.feet_contact_body_names)
        input_shape = self.session.input_shape
        output_shape = self.session.output_shape

        if len(input_shape) >= 2 and isinstance(input_shape[-1], int) and int(input_shape[-1]) != obs_dim:
            raise RuntimeError(
                f"TensorRT input obs dim mismatch: engine expects {int(input_shape[-1])}, controller builds {obs_dim}."
            )
        if len(output_shape) >= 2 and isinstance(output_shape[-1], int) and int(output_shape[-1]) != len(
            self.default_pos
        ):
            raise RuntimeError(
                "TensorRT output action dim mismatch: "
                f"engine outputs {int(output_shape[-1])}, robot has {len(self.default_pos)} DOFs."
            )

    def _is_self_hit(self, path: str | None) -> bool:
        return bool(path) and str(path).startswith(self.robot_prim_path)

    def _raycast_height(self, x: float, y: float, z_start: float) -> float | None:
        closest_distance = math.inf
        closest_hit_z = None

        def report_hit(hit) -> bool:
            nonlocal closest_distance, closest_hit_z

            rigid_body = str(getattr(hit, "rigid_body", "") or "")
            collision = str(getattr(hit, "collision", "") or "")
            if self._is_self_hit(rigid_body) or self._is_self_hit(collision):
                return True

            distance = float(getattr(hit, "distance", math.inf))
            if distance >= closest_distance:
                return True

            position = getattr(hit, "position", None)
            if position is None:
                return True

            closest_distance = distance
            closest_hit_z = float(position.z if hasattr(position, "z") else position[2])
            return True

        self.scene_query.raycast_all(
            carb.Float3(float(x), float(y), float(z_start)),
            carb.Float3(0.0, 0.0, -1.0),
            float(HEIGHT_SCAN_RAYCAST_DISTANCE),
            report_hit,
        )
        return closest_hit_z

    def _compute_height_scan(self, pos_w: np.ndarray, rot_wb: np.ndarray) -> np.ndarray:
        if len(self.height_scan_grid) == 0:
            return np.zeros(0, dtype=np.float32)

        yaw = math.atan2(float(rot_wb[1, 0]), float(rot_wb[0, 0]))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        yaw_rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
        world_xy = self.height_scan_grid @ yaw_rotation.T
        world_xy[:, 0] += float(pos_w[0])
        world_xy[:, 1] += float(pos_w[1])

        sensor_height = float(pos_w[2])
        ray_start_z = sensor_height + float(HEIGHT_SCAN_RAY_START_Z)
        height_scan = np.empty(len(world_xy), dtype=np.float32)

        fallback_height = np.clip(sensor_height - self.height_scan_offset, -1.0, 1.0)
        for index, (x_pos, y_pos) in enumerate(world_xy):
            hit_z = self._raycast_height(float(x_pos), float(y_pos), ray_start_z)
            if hit_z is None:
                height_scan[index] = fallback_height
            else:
                height_scan[index] = np.clip(sensor_height - hit_z - self.height_scan_offset, -1.0, 1.0)

        return height_scan

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        lin_vel_w = np.asarray(self.robot.get_linear_velocity(), dtype=np.float32)
        ang_vel_w = np.asarray(self.robot.get_angular_velocity(), dtype=np.float32)
        pos_w, quat_w = self.robot.get_world_pose()
        pos_w = np.asarray(pos_w, dtype=np.float32)
        quat_w = np.asarray(quat_w, dtype=np.float32)

        rot_wb = quat_to_rot_matrix(quat_w)
        rot_bw = rot_wb.transpose()
        lin_vel_b = rot_bw @ lin_vel_w
        ang_vel_b = rot_bw @ ang_vel_w
        gravity_b = rot_bw @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)

        joint_pos = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
        joint_vel = np.asarray(self.robot.get_joint_velocities(), dtype=np.float32)
        height_scan = self._compute_height_scan(pos_w, rot_wb)
        feet_contact_state = self._compute_feet_contact_state()

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
                feet_contact_state,
            ),
            dtype=np.float32,
        )

    def forward(self, step_index: int, command: np.ndarray):
        self._update_feet_contact_history()
        if step_index % self.decimation == 0:
            obs = self._compute_observation(command)
            outputs = self.session.run(obs.reshape(1, -1))
            self.action = np.asarray(outputs, dtype=np.float32).reshape(-1)
            self.previous_action = self.action.copy()

        target_positions = self.default_pos + (self.action * float(self.action_scale))
        self.robot.apply_action(ArticulationAction(joint_positions=target_positions))

    def shutdown(self):
        close_fn = getattr(self.session, "close", None)
        if callable(close_fn):
            close_fn()


G1TensorRtPolicyController = G1PolicyController
G1OnnxPolicyController = G1PolicyController
