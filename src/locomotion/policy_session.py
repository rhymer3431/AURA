"""Shared ONNX/TensorRT policy session helpers."""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort


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
                "TensorRT policy must expose exactly one input tensor and one output tensor. "
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
            raise RuntimeError(f"Dynamic TensorRT tensor shapes are not supported: {tensor_name}={shape}")
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
            raise RuntimeError("TensorRT execute_async_v3 failed.")

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


__all__ = ["create_policy_session", "infer_policy_backend"]
