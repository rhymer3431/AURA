from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .base import DetectorBackend, DetectorInfo
from .capabilities import DetectorSelection
from .stub_or_onnx_fallback import ColorSegFallbackConfig, ColorSegFallbackDetector
from .trt_yoloe import TensorRtYoloeDetector
from .ultralytics_yolo import UltralyticsYoloDetector


@dataclass(frozen=True)
class DetectorFactoryConfig:
    engine_path: str = ""
    model_path: str = ""
    device: str = ""
    fallback_label: str = "object"
    fallback_color: str = "red"


def default_engine_path() -> str:
    env_override = str(os.environ.get("ISAAC_AURA_YOLOE_ENGINE", "")).strip()
    if env_override != "":
        return str(Path(env_override).expanduser())
    return str(Path(__file__).resolve().parents[3] / "artifacts" / "models" / "yoloe-26s-seg-pf.engine")


def default_model_path() -> str:
    env_override = str(os.environ.get("ISAAC_AURA_YOLO_MODEL", "")).strip()
    if env_override != "":
        return str(Path(env_override).expanduser())
    return ""


def default_detector_device() -> str:
    return str(os.environ.get("ISAAC_AURA_YOLO_DEVICE", "")).strip()


def _resolve_detector_paths(config: DetectorFactoryConfig) -> tuple[str, str, str]:
    raw_engine_path = str(config.engine_path).strip()
    raw_model_path = str(config.model_path).strip()
    device = str(config.device).strip() or default_detector_device()

    if raw_model_path == "" and raw_engine_path == "":
        raw_model_path = default_model_path()
        if raw_model_path == "":
            raw_engine_path = default_engine_path()
    elif raw_model_path == "" and raw_engine_path != "" and Path(raw_engine_path).suffix.lower() != ".engine":
        raw_model_path = raw_engine_path
        raw_engine_path = ""

    engine_path = str(Path(raw_engine_path).expanduser()) if raw_engine_path != "" else ""
    model_path = str(Path(raw_model_path).expanduser()) if raw_model_path != "" else ""
    return model_path, engine_path, device


def select_detector_backend(config: DetectorFactoryConfig | None = None) -> DetectorSelection:
    cfg = config or DetectorFactoryConfig()
    model_path, engine_path, device = _resolve_detector_paths(cfg)

    if model_path != "":
        yolo_backend = UltralyticsYoloDetector(model_path, device=device)
        report = yolo_backend.probe()
        if yolo_backend.ready:
            report.selected_backend = yolo_backend.info.backend_name
            report.selected_reason = report.selected_reason or "ultralytics_backend_ready"
            return DetectorSelection(backend=yolo_backend, report=report)
        report.selected_backend = "color_seg_fallback"
        report.selected_reason = report.selected_reason or "fallback_required"
        fallback = ColorSegFallbackDetector(
            ColorSegFallbackConfig(default_label=cfg.fallback_label, color_name=cfg.fallback_color),
            warning=yolo_backend.info.warning,
            runtime_report=report,
            selected_reason=report.selected_reason,
        )
        return DetectorSelection(backend=fallback, report=report)

    engine_path = engine_path or default_engine_path()
    trt_backend = TensorRtYoloeDetector(engine_path)
    report = trt_backend.probe()
    if trt_backend.ready:
        report.selected_backend = trt_backend.info.backend_name
        report.selected_reason = report.selected_reason or "engine_ready"
        return DetectorSelection(backend=trt_backend, report=report)
    report.selected_backend = "color_seg_fallback"
    report.selected_reason = report.selected_reason or "fallback_required"
    fallback = ColorSegFallbackDetector(
        ColorSegFallbackConfig(default_label=cfg.fallback_label, color_name=cfg.fallback_color),
        warning=trt_backend.info.warning,
        runtime_report=report,
        selected_reason=report.selected_reason,
    )
    return DetectorSelection(backend=fallback, report=report)


def create_detector_backend(config: DetectorFactoryConfig | None = None) -> DetectorBackend:
    return select_detector_backend(config).backend


def describe_detector_backend(config: DetectorFactoryConfig | None = None) -> DetectorInfo:
    return create_detector_backend(config).info
