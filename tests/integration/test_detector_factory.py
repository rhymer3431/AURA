from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.detectors.factory import DetectorFactoryConfig, create_detector_backend, default_engine_path, select_detector_backend


def test_detector_factory_falls_back_cleanly_when_trt_backend_is_not_ready() -> None:
    backend = create_detector_backend(DetectorFactoryConfig(engine_path=default_engine_path(), fallback_label="apple"))

    assert backend.info.backend_name == "color_seg_fallback"
    assert backend.info.using_fallback is True
    assert backend.info.warning != ""


def test_detector_factory_returns_structured_runtime_report() -> None:
    selection = select_detector_backend(DetectorFactoryConfig(engine_path=default_engine_path(), fallback_label="apple"))

    assert selection.report.engine_exists is True
    assert selection.report.selected_backend in {"tensorrt_yoloe", "color_seg_fallback"}
    assert selection.report.selected_reason != ""
    if selection.report.selected_backend == "color_seg_fallback":
        assert selection.report.ready_for_inference is False
