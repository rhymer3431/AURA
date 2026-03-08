"""Robot and policy control logic."""

from __future__ import annotations

import fnmatch
import os
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


def _normalize_static_shape(shape: object) -> tuple[object, ...]:
    if isinstance(shape, tuple):
        return shape
    if isinstance(shape, list):
        return tuple(shape)
    return (shape,)


def infer_policy_backend(policy_path: str) -> str:
    suffix = os.path.splitext(policy_path)[1].lower()
    if suffix == ".engine":
        return "tensorrt"
    return "onnxruntime"


class _OnnxPolicySession:
    backend_name = "onnxruntime"

    def __init__(self, policy_path: str, providers: list[str]) -> None:
        self.session = ort.InferenceSession(policy_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = _normalize_static_shape(self.session.get_inputs()[0].shape)
        self.output_shape = _normalize_static_shape(self.session.get_outputs()[0].shape)

    def run(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.session.run([self.output_name], {self.input_name: observation.reshape(1, -1)})[0],
            dtype=np.float32,
        )

    def close(self) -> None:
        return None


class _TensorRtPolicySession:
    backend_name = "tensorrt"

    def __init__(self, policy_path: str, device_preference: str) -> None:
        if device_preference == "cpu":
            raise RuntimeError("TensorRT policy execution requires CUDA. Use --policy policy.onnx or a CUDA device.")

        try:
            from cuda.bindings import runtime as cudart
            import tensorrt as trt
        except ImportError as exc:
            try:
                from cuda import cudart
                import tensorrt as trt
            except ImportError:
                raise RuntimeError("TensorRT policy execution requires both `tensorrt` and `cuda-python`.") from exc

        self._cudart = cudart
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)

        with open(policy_path, "rb") as engine_file:
            serialized_engine = engine_file.read()

        self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {policy_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {policy_path}")

        input_names: list[str] = []
        output_names: list[str] = []
        for index in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(index)
            tensor_mode = self._engine.get_tensor_mode(tensor_name)
            if tensor_mode == trt.TensorIOMode.INPUT:
                input_names.append(tensor_name)
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                output_names.append(tensor_name)

        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                "G1 TensorRT policy must expose exactly one input tensor and one output tensor. "
                f"Found inputs={input_names}, outputs={output_names}."
            )

        self.input_name = input_names[0]
        self.output_name = output_names[0]
        self.input_shape = self._read_tensor_shape(self.input_name)
        self.output_shape = self._read_tensor_shape(self.output_name)
        self._input_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(self.input_name)))
        self._output_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(self.output_name)))
        self._input_bytes = int(np.prod(self.input_shape, dtype=np.int64)) * self._input_dtype.itemsize
        self._output_bytes = int(np.prod(self.output_shape, dtype=np.int64)) * self._output_dtype.itemsize

        self._input_ptr = 0
        self._output_ptr = 0
        self._stream = 0
        self._allocate_buffers()

    def _read_tensor_shape(self, tensor_name: str) -> tuple[int, ...]:
        shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(tensor_name))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(
                f"Dynamic TensorRT tensor shapes are not supported for the G1 policy engine: {tensor_name}={shape}"
            )
        return shape

    def _check_cuda(self, error_code: object, action: str) -> None:
        if error_code != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA call failed during {action}: {error_code}")

    def _allocate_buffers(self) -> None:
        try:
            error_code, stream = self._cudart.cudaStreamCreate()
            self._check_cuda(error_code, "cudaStreamCreate")
            self._stream = int(stream)

            error_code, input_ptr = self._cudart.cudaMalloc(self._input_bytes)
            self._check_cuda(error_code, "cudaMalloc(input)")
            self._input_ptr = int(input_ptr)

            error_code, output_ptr = self._cudart.cudaMalloc(self._output_bytes)
            self._check_cuda(error_code, "cudaMalloc(output)")
            self._output_ptr = int(output_ptr)
        except Exception:
            self.close()
            raise

    def run(self, observation: np.ndarray) -> np.ndarray:
        host_input = np.asarray(observation, dtype=self._input_dtype).reshape(self.input_shape)
        host_output = np.empty(self.output_shape, dtype=self._output_dtype)

        error_code, = self._cudart.cudaMemcpyAsync(
            self._input_ptr,
            host_input.ctypes.data,
            self._input_bytes,
            self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self._stream,
        )
        self._check_cuda(error_code, "cudaMemcpyAsync(HtoD)")

        if not self._context.set_tensor_address(self.input_name, self._input_ptr):
            raise RuntimeError(f"Failed to bind TensorRT input tensor: {self.input_name}")
        if not self._context.set_tensor_address(self.output_name, self._output_ptr):
            raise RuntimeError(f"Failed to bind TensorRT output tensor: {self.output_name}")
        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError("TensorRT execute_async_v3 failed for the G1 policy engine.")

        error_code, = self._cudart.cudaMemcpyAsync(
            host_output.ctypes.data,
            self._output_ptr,
            self._output_bytes,
            self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self._stream,
        )
        self._check_cuda(error_code, "cudaMemcpyAsync(DtoH)")

        error_code, = self._cudart.cudaStreamSynchronize(self._stream)
        self._check_cuda(error_code, "cudaStreamSynchronize")
        return np.asarray(host_output, dtype=np.float32)

    def close(self) -> None:
        if self._input_ptr:
            self._cudart.cudaFree(self._input_ptr)
            self._input_ptr = 0
        if self._output_ptr:
            self._cudart.cudaFree(self._output_ptr)
            self._output_ptr = 0
        if self._stream:
            self._cudart.cudaStreamDestroy(self._stream)
            self._stream = 0


def create_policy_session(policy_path: str, providers: list[str], device_preference: str):
    backend_name = infer_policy_backend(policy_path)
    if backend_name == "tensorrt":
        return _TensorRtPolicySession(policy_path=policy_path, device_preference=device_preference)
    return _OnnxPolicySession(policy_path=policy_path, providers=providers)


class G1PolicyController:
    """Standalone locomotion controller for the G1 robot."""

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        policy_path: str,
        position: np.ndarray,
        providers: list[str],
        device_preference: str,
        decimation: int,
    ) -> None:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import define_prim, get_prim_at_path

        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            prim.GetReferences().AddReference(usd_path)

        self.robot = SingleArticulation(prim_path=prim_path, name="g1", position=position)
        self.policy_session = create_policy_session(
            policy_path=policy_path,
            providers=providers,
            device_preference=device_preference,
        )
        self.input_name = self.policy_session.input_name
        self.output_name = self.policy_session.output_name
        self.backend_name = self.policy_session.backend_name
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

    def close(self) -> None:
        self.policy_session.close()

    def _validate_io(self):
        obs_dim = 12 + 3 * len(self.default_pos) + HEIGHT_SCAN_POINTS
        input_shape = self.policy_session.input_shape
        output_shape = self.policy_session.output_shape

        if len(input_shape) >= 2 and isinstance(input_shape[-1], int) and int(input_shape[-1]) != obs_dim:
            raise RuntimeError(
                f"Policy input obs dim mismatch: model expects {int(input_shape[-1])}, controller builds {obs_dim}."
            )
        if len(output_shape) >= 2 and isinstance(output_shape[-1], int) and int(output_shape[-1]) != len(
            self.default_pos
        ):
            raise RuntimeError(
                "Policy output action dim mismatch: "
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
            outputs = self.policy_session.run(obs)
            self.action = np.asarray(outputs, dtype=np.float32).reshape(-1)
            self.previous_action = self.action.copy()

        target_positions = self.default_pos + (self.action * ACTION_SCALE)
        self.robot.apply_action(ArticulationAction(joint_positions=target_positions))


G1OnnxPolicyController = G1PolicyController
