from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .base import DetectionResult, DetectorBackend, DetectorInfo
from .capabilities import DetectorRuntimeReport
from .postprocess.segmentation_utils import clamp_bbox_xyxy, mask_centroid


def is_cuda_device(device: str) -> bool:
    return str(device).lower().startswith("cuda")


def resolve_capture_device(infer_device: str, cuda_available: bool) -> str:
    if not cuda_available:
        return "cpu"

    normalized = str(infer_device).strip().lower()
    if normalized in {"", "auto"}:
        return "cuda:0"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "cuda:0"
    if normalized.startswith("cuda:"):
        first_cuda = normalized.split(",")[0].strip()
        return first_cuda if first_cuda != "cuda:" else "cuda:0"

    first = normalized.split(",")[0].strip()
    if first.isdigit():
        return f"cuda:{first}"
    if first.startswith("cuda:"):
        return first
    return "cpu"


class UltralyticsYoloDetector(DetectorBackend):
    def __init__(
        self,
        model_path: str,
        *,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        device: str = "",
        yolo_cls: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.infer_device = str(device).strip()
        self.capture_device = "cpu"
        self.tensor_device = torch.device("cpu")
        self.model_input_size = (self.imgsz, self.imgsz)
        self.model = None
        self.model_format = Path(self.model_path).suffix.lower()
        self._model_names: dict[int, str] = {}
        self._report = DetectorRuntimeReport(
            backend_name="ultralytics_yolo",
            engine_path=self.model_path,
            device=self.infer_device,
            model_format=self.model_format,
            selected_backend="ultralytics_yolo",
        )
        self._load(yolo_cls=yolo_cls)

    @property
    def info(self) -> DetectorInfo:
        warning = "; ".join(item for item in [*self._report.warnings, *self._report.errors] if item)
        return DetectorInfo(
            backend_name="ultralytics_yolo",
            engine_path=self.model_path,
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
        _ = timestamp
        if not self.ready:
            raise RuntimeError(self._format_runtime_error())
        image = np.asarray(rgb_image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"rgb_image must be [H,W,3], got {image.shape}")
        assert self.model is not None
        input_tensor = self._preprocess_rgb(image[..., :3])
        result = self.model.predict(
            source=input_tensor,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.infer_device,
            verbose=False,
        )[0]
        return self._decode_result(result, image_shape=image.shape[:2], metadata=metadata)

    def _load(self, *, yolo_cls: Callable[[str], Any] | None) -> None:
        model_path = Path(self.model_path)
        self._report.engine_exists = model_path.is_file()
        if not model_path.is_file():
            self._report.errors.append(f"model not found: {model_path}")
            self._report.selected_reason = "model_missing"
            return

        cuda_available = torch.cuda.is_available()
        if self.infer_device == "":
            self.infer_device = "0" if cuda_available else "cpu"

        infer_device_normalized = str(self.infer_device).strip().lower()
        if self.model_format == ".engine":
            if not cuda_available:
                self._report.errors.append("TensorRT engine requires CUDA, but CUDA is not available.")
                self._report.selected_reason = "cuda_unavailable"
                return
            if infer_device_normalized in {"", "auto"}:
                self.infer_device = "0"
                infer_device_normalized = "0"
            if infer_device_normalized == "cpu":
                self._report.errors.append("TensorRT engine cannot run on CPU. Use a CUDA device.")
                self._report.selected_reason = "engine_requires_cuda"
                return
            if not self._probe_tensorrt_engine(model_path):
                return

        self.capture_device = resolve_capture_device(self.infer_device, cuda_available)
        self.tensor_device = torch.device(self.capture_device if is_cuda_device(self.capture_device) else "cpu")
        self._report.device = self.infer_device

        if yolo_cls is None:
            try:
                from ultralytics import YOLO as yolo_cls
            except Exception as exc:  # noqa: BLE001
                self._report.errors.append(f"Ultralytics import failed: {type(exc).__name__}: {exc}")
                self._report.selected_reason = "ultralytics_import_failed"
                return

        try:
            self.model = yolo_cls(str(model_path))
        except Exception as exc:  # noqa: BLE001
            self._report.errors.append(f"YOLO model load failed: {type(exc).__name__}: {exc}")
            self._report.selected_reason = "model_load_failed"
            self.model = None
            return

        self._model_names = self._normalize_names(getattr(self.model, "names", {}))
        self._report.ready_for_inference = True
        self._report.selected_reason = "ultralytics_backend_ready"

    def _probe_tensorrt_engine(self, model_path: Path) -> bool:
        try:
            import tensorrt as trt

            self._report.tensorrt_import_ok = True
        except Exception as exc:  # noqa: BLE001
            self._report.errors.append(f"TensorRT import failed: {type(exc).__name__}: {exc}")
            self._report.selected_reason = "tensorrt_import_failed"
            return False

        runtime = None
        try:
            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            with open(model_path, "rb") as file_obj:
                engine = runtime.deserialize_cuda_engine(file_obj.read())
        except Exception as exc:  # noqa: BLE001
            message = f"TensorRT engine preflight failed: {type(exc).__name__}: {exc}"
            self._report.errors.append(message)
            self._report.selected_reason = "engine_incompatible"
            lowered = message.lower()
            if "serialization" in lowered or "platform" in lowered or "mismatch" in lowered:
                self._report.serialization_mismatch = True
            return False
        finally:
            del runtime

        if engine is None:
            self._report.errors.append("TensorRT engine preflight failed: deserialize returned None.")
            self._report.selected_reason = "engine_incompatible"
            self._report.serialization_mismatch = True
            return False

        self._report.deserialize_ok = True
        self._report.binding_metadata_ok = True
        return True

    def _preprocess_rgb(self, rgb_image: np.ndarray) -> torch.Tensor:
        rgb_tensor = torch.from_numpy(np.asarray(rgb_image, dtype=np.uint8)).to(
            device=self.tensor_device,
            non_blocking=True,
        )
        input_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        input_tensor = input_tensor.to(dtype=torch.float32)
        input_tensor /= 255.0

        if tuple(int(v) for v in input_tensor.shape[2:]) != self.model_input_size:
            input_tensor = F.interpolate(
                input_tensor,
                size=self.model_input_size,
                mode="bilinear",
                align_corners=False,
            )

        return input_tensor

    def _decode_result(
        self,
        result: Any,
        *,
        image_shape: tuple[int, int],
        metadata: dict[str, Any] | None,
    ) -> list[DetectionResult]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        box_tensor = getattr(boxes, "xyxy", None)
        if box_tensor is None:
            return []
        box_array = np.asarray(self._to_numpy(box_tensor), dtype=np.float32)
        if box_array.size == 0:
            return []
        box_array = np.atleast_2d(box_array)
        count = int(box_array.shape[0])

        conf_array = self._optional_vector(getattr(boxes, "conf", None), length=count, fill_value=1.0)
        cls_array = self._optional_vector(getattr(boxes, "cls", None), length=count, fill_value=0.0).astype(np.int64, copy=False)
        names = self._normalize_names(getattr(result, "names", {}))
        if not names:
            names = self._model_names
        masks = self._extract_masks(getattr(getattr(result, "masks", None), "data", None), image_shape=image_shape)
        output: list[DetectionResult] = []

        for index, bbox_xyxy in enumerate(box_array):
            scaled_bbox = self._scale_bbox_to_image(bbox_xyxy, image_shape=image_shape)
            bbox = clamp_bbox_xyxy(
                tuple(float(v) for v in scaled_bbox),
                width=int(image_shape[1]),
                height=int(image_shape[0]),
            )
            mask = None
            centroid = None
            if masks is not None and index < masks.shape[0]:
                mask = np.asarray(masks[index] >= 0.5, dtype=bool)
                centroid = mask_centroid(mask)
            if centroid is None:
                centroid = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

            class_id = int(cls_array[index]) if index < cls_array.shape[0] else 0
            class_name = self._resolve_class_name(class_id, names=names, metadata=metadata)
            detection_metadata: dict[str, Any] = {
                "backend": "ultralytics_yolo",
                "class_id": class_id,
                "model_path": self.model_path,
                "model_format": self.model_format,
            }
            if mask is not None:
                detection_metadata["mask_area_px"] = int(mask.sum())
            output.append(
                DetectionResult(
                    class_name=class_name,
                    confidence=float(conf_array[index]) if index < conf_array.shape[0] else 1.0,
                    bbox_xyxy=bbox,
                    mask=mask,
                    centroid_xy=centroid,
                    metadata=detection_metadata,
                )
            )
        return output

    def _scale_bbox_to_image(self, bbox_xyxy: np.ndarray, *, image_shape: tuple[int, int]) -> np.ndarray:
        height, width = int(image_shape[0]), int(image_shape[1])
        scale_x = float(width) / float(self.model_input_size[1])
        scale_y = float(height) / float(self.model_input_size[0])
        return np.asarray(
            [
                float(bbox_xyxy[0]) * scale_x,
                float(bbox_xyxy[1]) * scale_y,
                float(bbox_xyxy[2]) * scale_x,
                float(bbox_xyxy[3]) * scale_y,
            ],
            dtype=np.float32,
        )

    def _extract_masks(self, mask_data: Any, *, image_shape: tuple[int, int]) -> np.ndarray | None:
        if mask_data is None:
            return None
        tensor = torch.as_tensor(mask_data, dtype=torch.float32).detach().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim != 3:
            return None
        if tuple(int(v) for v in tensor.shape[-2:]) != tuple(int(v) for v in image_shape):
            tensor = F.interpolate(
                tensor.unsqueeze(1),
                size=tuple(int(v) for v in image_shape),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return np.asarray(tensor.numpy(), dtype=np.float32)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _optional_vector(value: Any, *, length: int, fill_value: float) -> np.ndarray:
        if value is None:
            return np.full((int(length),), float(fill_value), dtype=np.float32)
        array = np.asarray(UltralyticsYoloDetector._to_numpy(value), dtype=np.float32).reshape(-1)
        if array.shape[0] >= int(length):
            return array[: int(length)]
        padded = np.full((int(length),), float(fill_value), dtype=np.float32)
        padded[: array.shape[0]] = array
        return padded

    @staticmethod
    def _normalize_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
        return {}

    @staticmethod
    def _resolve_class_name(class_id: int, *, names: dict[int, str], metadata: dict[str, Any] | None) -> str:
        if class_id in names and str(names[class_id]).strip() != "":
            return str(names[class_id])
        fallback = str((metadata or {}).get("target_class_hint", "")).strip()
        if fallback != "":
            return fallback
        return f"class_{class_id}"

    def _format_runtime_error(self) -> str:
        issues = [*self._report.errors, *self._report.warnings]
        if not issues:
            return f"{self.info.backend_name} is not ready"
        return "; ".join(str(item) for item in issues if str(item).strip() != "")
