from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import DetectionResult, DetectorBackend, DetectorInfo


class TensorRtYoloeDetector(DetectorBackend):
    def __init__(self, engine_path: str) -> None:
        self.engine_path = str(Path(engine_path))
        self._runtime_error = ""
        self._runtime = None
        self._engine = None
        self._context = None
        self._load_engine()

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(
            backend_name="tensorrt_yoloe",
            engine_path=self.engine_path,
            warning=self._runtime_error,
            using_fallback=False,
        )

    @property
    def ready(self) -> bool:
        return self._context is not None and self._runtime_error == ""

    def detect(
        self,
        rgb_image: np.ndarray,
        *,
        timestamp: float,
        metadata: dict[str, Any] | None = None,
    ) -> list[DetectionResult]:
        _ = rgb_image, timestamp, metadata
        if not self.ready:
            raise RuntimeError(self._runtime_error or "TensorRT detector is not ready.")
        raise NotImplementedError(
            "YOLOE TensorRT post-processing is not implemented yet. "
            "Factory should fall back when this backend is not fully ready."
        )

    def _load_engine(self) -> None:
        engine_path = Path(self.engine_path)
        if not engine_path.exists():
            self._runtime_error = f"engine not found: {engine_path}"
            return
        try:
            import tensorrt as trt
        except Exception as exc:  # noqa: BLE001
            self._runtime_error = f"TensorRT import failed: {type(exc).__name__}: {exc}"
            return
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
            self._runtime_error = (
                "TensorRT engine loaded, but YOLOE output decode path still needs binding-specific "
                "post-processing. Falling back is recommended until that is implemented."
            )
            self._context = None
        except Exception as exc:  # noqa: BLE001
            self._runtime_error = f"TensorRT engine load failed: {type(exc).__name__}: {exc}"
