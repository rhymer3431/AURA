from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.detectors.factory import DetectorFactoryConfig, create_detector_backend, default_engine_path


def test_detector_factory_falls_back_cleanly_when_trt_backend_is_not_ready() -> None:
    backend = create_detector_backend(DetectorFactoryConfig(engine_path=default_engine_path(), fallback_label="apple"))

    assert backend.info.backend_name == "color_seg_fallback"
    assert backend.info.using_fallback is True
    assert backend.info.warning != ""
