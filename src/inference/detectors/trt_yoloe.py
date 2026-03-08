from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.cv2_compat import cv2

from .base import DetectionResult, DetectorBackend, DetectorInfo
from .capabilities import BindingInfo, DetectorRuntimeReport
from .postprocess.yoloe_decode import decode_yoloe_predictions


class TensorRtYoloeDetector(DetectorBackend):
    def __init__(
        self,
        engine_path: str,
        *,
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.25,
        mask_dim: int = 32,
    ) -> None:
        self.engine_path = str(Path(engine_path).expanduser())
        self.class_names = list(class_names or [])
        self.confidence_threshold = float(confidence_threshold)
        self.mask_dim = int(mask_dim)
        self._trt = None
        self._cudart = None
        self._runtime = None
        self._engine = None
        self._context = None
        self._input_bindings: list[BindingInfo] = []
        self._output_bindings: list[BindingInfo] = []
        self._report = DetectorRuntimeReport(
            backend_name="tensorrt_yoloe",
            engine_path=self.engine_path,
            device="cuda",
            model_format=Path(self.engine_path).suffix.lower(),
            selected_backend="tensorrt_yoloe",
        )
        self._probe()

    @property
    def info(self) -> DetectorInfo:
        warning = "; ".join(item for item in [*self._report.warnings, *self._report.errors] if item)
        return DetectorInfo(
            backend_name="tensorrt_yoloe",
            engine_path=self.engine_path,
            warning=warning,
            using_fallback=False,
            selected_reason=self._report.selected_reason,
            runtime_report=self._report,
        )

    @property
    def ready(self) -> bool:
        return bool(self._report.ready_for_inference)

    def probe(self) -> DetectorRuntimeReport:
        return self._report

    def detect(
        self,
        rgb_image: np.ndarray,
        *,
        timestamp: float,
        metadata: dict[str, Any] | None = None,
    ) -> list[DetectionResult]:
        raw_outputs = self.detect_raw(rgb_image, metadata=metadata)
        decoded = self.decode(raw_outputs, image_shape=np.asarray(rgb_image).shape[:2], metadata=metadata)
        results: list[DetectionResult] = []
        for item in decoded:
            enriched_metadata = {
                **item.metadata,
                "timestamp": float(timestamp),
            }
            results.append(
                DetectionResult(
                    class_name=item.class_name,
                    confidence=float(item.confidence),
                    bbox_xyxy=item.bbox_xyxy,
                    mask=item.mask,
                    centroid_xy=item.centroid_xy,
                    metadata=enriched_metadata,
                )
            )
        return results

    def detect_raw(
        self,
        rgb_image: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        _ = metadata
        if not self.ready:
            raise RuntimeError(self._format_runtime_error())
        image = np.asarray(rgb_image, dtype=np.uint8)
        input_info = self._input_bindings[0]
        input_tensor = self._preprocess(image, input_shape=input_info.shape)
        return self._execute(input_tensor)

    def decode(
        self,
        raw_outputs: dict[str, np.ndarray],
        *,
        image_shape: tuple[int, int],
        metadata: dict[str, Any] | None = None,
    ):
        metadata = dict(metadata or {})
        predictions = None
        proto = None
        for name, array in raw_outputs.items():
            lowered = name.lower()
            if predictions is None and any(token in lowered for token in ("pred", "boxes", "det")):
                predictions = array
            if proto is None and "proto" in lowered:
                proto = array
        if predictions is None:
            predictions = next(iter(raw_outputs.values()))
        class_names = metadata.get("class_names")
        names = list(class_names) if isinstance(class_names, list) else self.class_names
        if not names:
            names = [str(metadata.get("target_class_hint", "object"))]
        return decode_yoloe_predictions(
            predictions,
            image_shape=image_shape,
            num_classes=max(len(names), 1),
            class_names=names,
            confidence_threshold=self.confidence_threshold,
            proto=proto,
            mask_dim=self.mask_dim if proto is not None else 0,
        )

    def _probe(self) -> None:
        engine_path = Path(self.engine_path)
        self._report.engine_exists = engine_path.exists()
        if not engine_path.exists():
            self._report.errors.append(f"engine not found: {engine_path}")
            self._report.selected_reason = "engine_missing"
            return
        try:
            import tensorrt as trt
        except Exception as exc:  # noqa: BLE001
            self._report.errors.append(f"TensorRT import failed: {type(exc).__name__}: {exc}")
            self._report.selected_reason = "tensorrt_import_failed"
            return
        self._trt = trt
        self._report.tensorrt_import_ok = True

        try:
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
            if engine is None:
                raise RuntimeError("deserialize_cuda_engine returned None")
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError("create_execution_context returned None")
            self._runtime = runtime
            self._engine = engine
            self._context = context
            self._report.deserialize_ok = True
        except Exception as exc:  # noqa: BLE001
            message = f"TensorRT engine load failed: {type(exc).__name__}: {exc}"
            self._report.errors.append(message)
            lowered = message.lower()
            if "serialization" in lowered or "version" in lowered or "magictag" in lowered:
                self._report.serialization_mismatch = True
                self._report.selected_reason = "serialization_mismatch"
            else:
                self._report.selected_reason = "deserialize_failed"
            self._context = None
            return

        try:
            self._input_bindings, self._output_bindings = self._collect_bindings()
            self._report.inputs = list(self._input_bindings)
            self._report.outputs = list(self._output_bindings)
            self._report.binding_metadata_ok = bool(self._input_bindings and self._output_bindings)
        except Exception as exc:  # noqa: BLE001
            self._report.errors.append(f"binding metadata failed: {type(exc).__name__}: {exc}")
            self._report.selected_reason = "binding_metadata_failed"
            self._context = None
            return

        try:
            from cuda import cudart  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._report.warnings.append(f"CUDA runtime import failed: {type(exc).__name__}: {exc}")
            self._report.selected_reason = "cuda_runtime_missing"
            self._context = None
            return

        self._cudart = cudart
        self._report.ready_for_inference = True
        self._report.selected_reason = "trt_backend_ready"

    def _collect_bindings(self) -> tuple[list[BindingInfo], list[BindingInfo]]:
        assert self._engine is not None
        inputs: list[BindingInfo] = []
        outputs: list[BindingInfo] = []
        if hasattr(self._engine, "num_io_tensors"):
            for index in range(int(self._engine.num_io_tensors)):
                name = str(self._engine.get_tensor_name(index))
                mode_obj = self._engine.get_tensor_mode(name)
                mode_name = "input" if "input" in str(mode_obj).lower() else "output"
                shape = tuple(int(value) for value in self._engine.get_tensor_shape(name))
                dtype = str(self._engine.get_tensor_dtype(name))
                info = BindingInfo(name=name, index=index, shape=shape, dtype=dtype, mode=mode_name)
                if mode_name == "input":
                    inputs.append(info)
                else:
                    outputs.append(info)
            return inputs, outputs

        for index in range(int(self._engine.num_bindings)):
            name = str(self._engine.get_binding_name(index))
            mode_name = "input" if bool(self._engine.binding_is_input(index)) else "output"
            shape = tuple(int(value) for value in self._engine.get_binding_shape(index))
            dtype = str(self._engine.get_binding_dtype(index))
            info = BindingInfo(name=name, index=index, shape=shape, dtype=dtype, mode=mode_name)
            if mode_name == "input":
                inputs.append(info)
            else:
                outputs.append(info)
        return inputs, outputs

    def _preprocess(self, image: np.ndarray, *, input_shape: tuple[int, ...]) -> np.ndarray:
        if len(input_shape) < 4:
            raise RuntimeError(f"Unsupported input shape: {input_shape}")
        _, channels, height, width = input_shape[-4:]
        if channels != 3:
            raise RuntimeError(f"Unsupported channel count for YOLOE engine: {channels}")
        resized = cv2.resize(np.asarray(image, dtype=np.uint8), (int(width), int(height)), interpolation=cv2.INTER_AREA)
        tensor = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor.astype(np.float32))

    def _execute(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        assert self._engine is not None
        assert self._context is not None
        assert self._cudart is not None
        cudart = self._cudart
        allocations: list[int] = []
        try:
            output_specs = self._resolve_output_specs(input_tensor)
            device_buffers: dict[str, int] = {}
            host_outputs: dict[str, np.ndarray] = {}
            if hasattr(self._context, "set_input_shape"):
                self._context.set_input_shape(self._input_bindings[0].name, input_tensor.shape)

            for binding in self._input_bindings:
                device_ptr = self._cuda_malloc(int(input_tensor.nbytes))
                allocations.append(device_ptr)
                self._cuda_memcpy_htod(device_ptr, input_tensor)
                device_buffers[binding.name] = device_ptr
                if hasattr(self._context, "set_tensor_address"):
                    self._context.set_tensor_address(binding.name, int(device_ptr))
            for output_name, output_shape in output_specs.items():
                output = np.empty(output_shape, dtype=np.float32)
                device_ptr = self._cuda_malloc(int(output.nbytes))
                allocations.append(device_ptr)
                device_buffers[output_name] = device_ptr
                host_outputs[output_name] = output
                if hasattr(self._context, "set_tensor_address"):
                    self._context.set_tensor_address(output_name, int(device_ptr))

            if hasattr(self._context, "execute_async_v3"):
                success = bool(self._context.execute_async_v3(0))
            else:
                bindings = [0] * self._engine.num_bindings
                for binding in self._input_bindings:
                    bindings[binding.index] = int(device_buffers[binding.name])
                for binding in self._output_bindings:
                    bindings[binding.index] = int(device_buffers[binding.name])
                success = bool(self._context.execute_v2(bindings))
            if not success:
                raise RuntimeError("TensorRT execution returned false")

            for output_name, output in host_outputs.items():
                self._cuda_memcpy_dtoh(output, device_buffers[output_name])
            return host_outputs
        finally:
            for pointer in reversed(allocations):
                try:
                    cudart.cudaFree(pointer)
                except Exception:  # noqa: BLE001
                    continue

    def _resolve_output_specs(self, input_tensor: np.ndarray) -> dict[str, tuple[int, ...]]:
        assert self._engine is not None
        assert self._context is not None
        specs: dict[str, tuple[int, ...]] = {}
        if hasattr(self._context, "set_input_shape"):
            self._context.set_input_shape(self._input_bindings[0].name, input_tensor.shape)
            for binding in self._output_bindings:
                shape = tuple(int(value) for value in self._context.get_tensor_shape(binding.name))
                specs[binding.name] = tuple(max(1, dim) for dim in shape)
            return specs

        if hasattr(self._context, "set_binding_shape"):
            self._context.set_binding_shape(self._input_bindings[0].index, tuple(int(value) for value in input_tensor.shape))
            for binding in self._output_bindings:
                shape = tuple(int(value) for value in self._context.get_binding_shape(binding.index))
                specs[binding.name] = tuple(max(1, dim) for dim in shape)
            return specs

        for binding in self._output_bindings:
            specs[binding.name] = tuple(max(1, dim) for dim in binding.shape)
        return specs

    def _cuda_malloc(self, size_bytes: int) -> int:
        assert self._cudart is not None
        status, device_ptr = self._cudart.cudaMalloc(int(size_bytes))
        if int(status) != 0:
            raise RuntimeError(f"cudaMalloc failed with status={status}")
        return int(device_ptr)

    def _cuda_memcpy_htod(self, device_ptr: int, host: np.ndarray) -> None:
        assert self._cudart is not None
        host_arr = np.ascontiguousarray(host)
        status = self._cudart.cudaMemcpy(
            int(device_ptr),
            int(host_arr.ctypes.data),
            int(host_arr.nbytes),
            self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )[0]
        if int(status) != 0:
            raise RuntimeError(f"cudaMemcpy HtoD failed with status={status}")

    def _cuda_memcpy_dtoh(self, host: np.ndarray, device_ptr: int) -> None:
        assert self._cudart is not None
        host_arr = np.ascontiguousarray(host)
        status = self._cudart.cudaMemcpy(
            int(host_arr.ctypes.data),
            int(device_ptr),
            int(host_arr.nbytes),
            self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )[0]
        if int(status) != 0:
            raise RuntimeError(f"cudaMemcpy DtoH failed with status={status}")
        np.copyto(host, host_arr)

    def _format_runtime_error(self) -> str:
        return "; ".join([*self._report.warnings, *self._report.errors]) or "TensorRT detector is not ready."
