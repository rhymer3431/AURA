from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .base import DetectorBackend, DetectorInfo
from .capabilities import DetectorRuntimeReport, DetectorSelection
from .stub_or_onnx_fallback import ColorSegFallbackConfig, ColorSegFallbackDetector
from .ultralytics_yolo import UltralyticsYoloDetector

DEFAULT_ULTRALYTICS_MODEL_NAME = "yolo26s.pt"


@dataclass(frozen=True)
class DetectorFactoryConfig:
    model_path: str = ""
    device: str = ""
    fallback_label: str = "object"
    fallback_color: str = "red"


def default_model_candidates(repo_root: Path | None = None) -> list[Path]:
    root = repo_root or Path(__file__).resolve().parents[3]
    return [
        root / "artifacts" / "models" / DEFAULT_ULTRALYTICS_MODEL_NAME,
        root.parent / "isaac" / DEFAULT_ULTRALYTICS_MODEL_NAME,
    ]


def default_model_path(repo_root: Path | None = None) -> str:
    env_override = str(os.environ.get("ISAAC_AURA_YOLO_MODEL", "")).strip()
    if env_override != "":
        return str(Path(env_override).expanduser())
    for candidate in default_model_candidates(repo_root):
        if candidate.is_file():
            return str(candidate.resolve())
    return ""


def default_detector_device() -> str:
    return str(os.environ.get("ISAAC_AURA_YOLO_DEVICE", "")).strip()


def _resolve_detector_paths(config: DetectorFactoryConfig) -> tuple[str, str]:
    raw_model_path = str(config.model_path).strip()

    if raw_model_path == "":
        raw_model_path = default_model_path()

    model_path = str(Path(raw_model_path).expanduser()) if raw_model_path != "" else ""
    device = str(config.device).strip() or default_detector_device()
    return model_path, device


def select_detector_backend(config: DetectorFactoryConfig | None = None) -> DetectorSelection:
    cfg = config or DetectorFactoryConfig()
    model_path, device = _resolve_detector_paths(cfg)

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

    fallback_report = DetectorRuntimeReport(
        backend_name="ultralytics_yolo",
        engine_path="",
        device=device,
        selected_backend="color_seg_fallback",
        selected_reason="model_missing",
        errors=["model not found: no detector model path configured"],
    )
    fallback_reason = fallback_report.selected_reason or "model_missing"
    fallback_warning = "; ".join(fallback_report.errors)
    fallback = ColorSegFallbackDetector(
        ColorSegFallbackConfig(default_label=cfg.fallback_label, color_name=cfg.fallback_color),
        warning=fallback_warning,
        runtime_report=fallback_report,
        selected_reason=fallback_reason,
    )
    return DetectorSelection(backend=fallback, report=fallback_report)


def create_detector_backend(config: DetectorFactoryConfig | None = None) -> DetectorBackend:
    return select_detector_backend(config).backend


def describe_detector_backend(config: DetectorFactoryConfig | None = None) -> DetectorInfo:
    return create_detector_backend(config).info
