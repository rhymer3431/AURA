"""Robot and policy control logic."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from common.geometry import quat_wxyz_to_yaw

from .constants import (
    ACTION_SCALE,
    DAMPING_PATTERNS,
    DEFAULT_JOINT_POS_PATTERNS,
    HEIGHT_SCAN_GRID_LENGTH,
    HEIGHT_SCAN_GRID_RESOLUTION,
    HEIGHT_SCAN_GRID_WIDTH,
    HEIGHT_SCAN_OFFSET,
    HEIGHT_SCAN_POINTS,
    HEIGHT_SCAN_RAY_DISTANCE,
    HEIGHT_SCAN_RAY_START_Z,
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


def _build_height_scan_grid() -> np.ndarray:
    x_count = int(round(HEIGHT_SCAN_GRID_LENGTH / HEIGHT_SCAN_GRID_RESOLUTION)) + 1
    y_count = int(round(HEIGHT_SCAN_GRID_WIDTH / HEIGHT_SCAN_GRID_RESOLUTION)) + 1
    x_coords = np.linspace(-0.5 * HEIGHT_SCAN_GRID_LENGTH, 0.5 * HEIGHT_SCAN_GRID_LENGTH, num=x_count, dtype=np.float32)
    y_coords = np.linspace(-0.5 * HEIGHT_SCAN_GRID_WIDTH, 0.5 * HEIGHT_SCAN_GRID_WIDTH, num=y_count, dtype=np.float32)
    grid = np.asarray([[x, y] for y in y_coords for x in x_coords], dtype=np.float32)
    if grid.shape != (HEIGHT_SCAN_POINTS, 2):
        raise RuntimeError(f"Height scan grid mismatch: expected {(HEIGHT_SCAN_POINTS, 2)}, got {grid.shape}.")
    return grid


def _extract_raycast_hit_position(hit_result: object) -> np.ndarray | None:
    if isinstance(hit_result, tuple) and len(hit_result) == 2 and isinstance(hit_result[0], bool):
        if not hit_result[0]:
            return None
        return _extract_raycast_hit_position(hit_result[1])

    if isinstance(hit_result, dict):
        if not bool(hit_result.get("hit", False)):
            return None
        if "position" in hit_result:
            pos = np.asarray(hit_result["position"], dtype=np.float32).reshape(-1)
            if pos.shape[0] >= 3:
                return pos[:3].copy()
        return None

    if hit_result is None:
        return None

    for attr_name in ("position", "point", "hit_position"):
        pos = getattr(hit_result, attr_name, None)
        if pos is None:
            continue
        pos_np = np.asarray(pos, dtype=np.float32).reshape(-1)
        if pos_np.shape[0] >= 3:
            return pos_np[:3].copy()

    if hasattr(hit_result, "hit") and not bool(getattr(hit_result, "hit")):
        return None
    return None


def _extract_raycast_hit_path(hit_result: object) -> str:
    if isinstance(hit_result, tuple) and len(hit_result) == 2 and isinstance(hit_result[0], bool):
        if not hit_result[0]:
            return ""
        return _extract_raycast_hit_path(hit_result[1])

    if isinstance(hit_result, dict):
        for key in ("collision", "collisionPath", "rigid_body", "rigidBody", "collider", "body"):
            value = hit_result.get(key)
            if value:
                return str(value)
        return ""

    if hit_result is None:
        return ""

    for attr_name in ("collision", "collisionPath", "rigid_body", "rigidBody", "collider", "body"):
        value = getattr(hit_result, attr_name, None)
        if value:
            return str(value)
    return ""


def _path_has_ground_token(path: str) -> bool:
    normalized = str(path).replace("\\", "/").strip().lower()
    if normalized == "":
        return False
    tokens = [token for token in normalized.replace("-", "_").split("/") if token]
    ground_keywords = ("ground", "terrain", "floor", "plane", "walkway", "walkable")
    return any(any(keyword in token for keyword in ground_keywords) for token in tokens)


class _HeightScanner:
    """Standalone replacement for the Isaac Lab height scanner observation."""

    def __init__(self, robot_prim_path: str) -> None:
        self.robot_prim_path = str(robot_prim_path).rstrip("/")
        self.sensor_prim_path = f"{self.robot_prim_path}/torso_link"
        self.pattern_xy = _build_height_scan_grid()
        self._query_interface = None
        self._vector_type = None
        self._allowed_hit_path_prefixes: tuple[str, ...] | None = None
        self._last_scan: np.ndarray | None = None
        self._warned_ground_filter = False

    def _resolve_query_interface(self):
        if self._query_interface is not None:
            return self._query_interface

        try:
            from omni.physx import get_physx_scene_query_interface
        except Exception:  # noqa: BLE001
            get_physx_scene_query_interface = None

        if get_physx_scene_query_interface is not None:
            try:
                self._query_interface = get_physx_scene_query_interface()
                return self._query_interface
            except Exception:  # noqa: BLE001
                self._query_interface = None

        try:
            from omni.physics.core import get_physics_scene_query_interface
        except Exception:  # noqa: BLE001
            get_physics_scene_query_interface = None

        if get_physics_scene_query_interface is not None:
            self._query_interface = get_physics_scene_query_interface()
        return self._query_interface

    def _vector(self, xyz: np.ndarray):
        if self._vector_type is False:
            return tuple(float(v) for v in xyz)
        if self._vector_type is None:
            try:
                import carb

                self._vector_type = carb.Float3
            except Exception:  # noqa: BLE001
                self._vector_type = False
        if self._vector_type is False:
            return tuple(float(v) for v in xyz)
        return self._vector_type(float(xyz[0]), float(xyz[1]), float(xyz[2]))

    @staticmethod
    def _prim_pose(stage, prim_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            from pxr import Usd, UsdGeom
        except Exception:  # noqa: BLE001
            return None, None

        try:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return None, None
            xform = UsdGeom.Xformable(prim)
            world_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            trans = world_tf.ExtractTranslation()
            quat_gf = world_tf.ExtractRotationQuat()
            imag = quat_gf.GetImaginary()
            pos = np.asarray([trans[0], trans[1], trans[2]], dtype=np.float32)
            quat = np.asarray([quat_gf.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float32)
            return pos, quat
        except Exception:  # noqa: BLE001
            return None, None

    def _sensor_pose(self, base_state: RobotBaseState) -> tuple[np.ndarray, float]:
        try:
            import omni.usd
        except Exception:  # noqa: BLE001
            return base_state.position_w[:3].copy(), float(quat_wxyz_to_yaw(base_state.quat_wxyz))

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return base_state.position_w[:3].copy(), float(quat_wxyz_to_yaw(base_state.quat_wxyz))

        pos, quat = self._prim_pose(stage, self.sensor_prim_path)
        if pos is None or quat is None:
            pos, quat = self._prim_pose(stage, self.robot_prim_path)
        if pos is None or quat is None:
            return base_state.position_w[:3].copy(), float(quat_wxyz_to_yaw(base_state.quat_wxyz))
        return pos, float(quat_wxyz_to_yaw(quat))

    def _resolve_allowed_hit_path_prefixes(self) -> tuple[str, ...]:
        if self._allowed_hit_path_prefixes is not None:
            return self._allowed_hit_path_prefixes

        prefixes: set[str] = {"/World/ground", "/World/defaultGroundPlane"}
        try:
            import omni.usd
        except Exception:  # noqa: BLE001
            self._allowed_hit_path_prefixes = tuple(sorted(prefixes))
            return self._allowed_hit_path_prefixes

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            try:
                for prim in stage.Traverse():
                    prim_path = str(prim.GetPath())
                    if _path_has_ground_token(prim_path):
                        prefixes.add(prim_path)
            except Exception:  # noqa: BLE001
                pass

        self._allowed_hit_path_prefixes = tuple(sorted(prefixes))
        return self._allowed_hit_path_prefixes

    def _is_allowed_hit(self, hit_result: object) -> bool:
        hit_path = _extract_raycast_hit_path(hit_result)
        if hit_path.startswith(self.robot_prim_path):
            return False
        prefixes = self._resolve_allowed_hit_path_prefixes()
        if hit_path == "":
            return False
        return any(hit_path.startswith(prefix) for prefix in prefixes)

    def _raycast(self, origin: np.ndarray):
        scene_query = self._resolve_query_interface()
        if scene_query is None:
            return None
        direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        first_valid_hit = None

        def _report(hit) -> bool:
            nonlocal first_valid_hit
            if not self._is_allowed_hit(hit):
                return True
            first_valid_hit = hit
            return False

        try:
            scene_query.raycast_all(self._vector(origin), self._vector(direction), HEIGHT_SCAN_RAY_DISTANCE, _report)
            if first_valid_hit is not None:
                return first_valid_hit
        except TypeError:
            try:
                scene_query.raycast_all(tuple(float(v) for v in origin), (0.0, 0.0, -1.0), HEIGHT_SCAN_RAY_DISTANCE, _report)
                if first_valid_hit is not None:
                    return first_valid_hit
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

        try:
            hit = scene_query.raycast_closest(self._vector(origin), self._vector(direction), HEIGHT_SCAN_RAY_DISTANCE)
            return hit if self._is_allowed_hit(hit) else None
        except TypeError:
            hit = scene_query.raycast_closest(tuple(float(v) for v in origin), (0.0, 0.0, -1.0), HEIGHT_SCAN_RAY_DISTANCE)
            return hit if self._is_allowed_hit(hit) else None
        except Exception:  # noqa: BLE001
            return None

    def fallback_scan(self, base_state: RobotBaseState) -> np.ndarray:
        height_value = np.clip(float(base_state.position_w[2]) - HEIGHT_SCAN_OFFSET, -1.0, 1.0)
        return np.full(HEIGHT_SCAN_POINTS, height_value, dtype=np.float32)

    def scan(self, base_state: RobotBaseState) -> np.ndarray:
        if self._resolve_query_interface() is None:
            return self.fallback_scan(base_state)

        allowed_prefixes = self._resolve_allowed_hit_path_prefixes()
        if len(allowed_prefixes) == 0:
            return self._last_scan.copy() if self._last_scan is not None else self.fallback_scan(base_state)

        sensor_pos_w, sensor_yaw = self._sensor_pose(base_state)
        cos_yaw = float(np.cos(sensor_yaw))
        sin_yaw = float(np.sin(sensor_yaw))

        scan = np.empty(HEIGHT_SCAN_POINTS, dtype=np.float32)
        valid_hits = 0
        origin_xy = np.empty((HEIGHT_SCAN_POINTS, 2), dtype=np.float32)
        origin_xy[:, 0] = sensor_pos_w[0] + (self.pattern_xy[:, 0] * cos_yaw) - (self.pattern_xy[:, 1] * sin_yaw)
        origin_xy[:, 1] = sensor_pos_w[1] + (self.pattern_xy[:, 0] * sin_yaw) + (self.pattern_xy[:, 1] * cos_yaw)

        for idx in range(HEIGHT_SCAN_POINTS):
            origin = np.asarray([origin_xy[idx, 0], origin_xy[idx, 1], sensor_pos_w[2] + HEIGHT_SCAN_RAY_START_Z], dtype=np.float32)
            hit_pos = _extract_raycast_hit_position(self._raycast(origin))
            if hit_pos is None:
                scan[idx] = 1.0
                continue
            valid_hits += 1
            scan[idx] = float(sensor_pos_w[2]) - float(hit_pos[2]) - HEIGHT_SCAN_OFFSET
        if valid_hits < max(1, int(0.6 * HEIGHT_SCAN_POINTS)):
            if not self._warned_ground_filter:
                print("[WARN] Height scan found too few ground hits; using fallback scan for stability.")
                self._warned_ground_filter = True
            return self._last_scan.copy() if self._last_scan is not None else self.fallback_scan(base_state)
        clipped = np.clip(scan, -1.0, 1.0)
        self._last_scan = clipped.copy()
        return clipped


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

        self._engine = self._deserialize_engine(policy_path)
        if self._engine is None:
            print(f"[ERROR] TensorRT engine is incompatible or unreadable: {policy_path}")
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

    def _deserialize_engine(self, engine_path: str):
        with open(engine_path, "rb") as engine_file:
            serialized_engine = engine_file.read()
        return self._runtime.deserialize_cuda_engine(serialized_engine)

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
        self.prim_path = prim_path

        self.default_pos = np.zeros(0, dtype=np.float32)
        self.default_vel = np.zeros(0, dtype=np.float32)
        self.stiffness = np.zeros(0, dtype=np.float32)
        self.damping = np.zeros(0, dtype=np.float32)
        self.max_effort = np.zeros(0, dtype=np.float32)
        self.max_velocity = np.zeros(0, dtype=np.float32)

        self.action = np.zeros(0, dtype=np.float32)
        self.previous_action = np.zeros(0, dtype=np.float32)
        self.policy_counter = 0
        self.height_scanner: _HeightScanner | None = None

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
        self.height_scanner = _HeightScanner(robot_prim_path=self.prim_path)
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
        if self.height_scanner is None:
            height_scan = np.full(HEIGHT_SCAN_POINTS, np.clip(float(pos_w[2]) - HEIGHT_SCAN_OFFSET, -1.0, 1.0), dtype=np.float32)
        else:
            height_scan = self.height_scanner.scan(base_state)

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
