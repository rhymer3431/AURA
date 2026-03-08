from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.detectors.factory import (
    DEFAULT_ULTRALYTICS_MODEL_NAME,
    DetectorFactoryConfig,
    create_detector_backend,
    default_engine_path,
    default_model_path,
    select_detector_backend,
)


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


def test_detector_factory_uses_explicit_model_path_before_trt_and_falls_back_cleanly(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.pt"

    selection = select_detector_backend(DetectorFactoryConfig(model_path=str(missing_model), fallback_label="apple"))

    assert selection.backend.info.backend_name == "color_seg_fallback"
    assert selection.report.backend_name == "ultralytics_yolo"
    assert selection.report.selected_backend == "color_seg_fallback"
    assert selection.report.selected_reason == "model_missing"


def test_detector_factory_treats_non_engine_path_as_model_alias(tmp_path: Path) -> None:
    missing_model = tmp_path / "alias.pt"

    selection = select_detector_backend(DetectorFactoryConfig(engine_path=str(missing_model), fallback_label="apple"))

    assert selection.backend.info.backend_name == "color_seg_fallback"
    assert selection.report.backend_name == "ultralytics_yolo"
    assert selection.report.engine_path == str(missing_model)
    assert selection.report.selected_reason == "model_missing"


def test_default_model_path_uses_reference_yolo_engine_candidate_when_present(tmp_path: Path) -> None:
    repo_root = tmp_path / "isaac-aura"
    sibling_project = tmp_path / "isaac"
    repo_root.mkdir()
    sibling_project.mkdir()
    model_path = sibling_project / DEFAULT_ULTRALYTICS_MODEL_NAME
    model_path.write_text("stub", encoding="utf-8")

    assert default_model_path(repo_root=repo_root) == str(model_path.resolve())


def test_default_model_path_prefers_repo_artifact_engine_before_sibling_candidate(tmp_path: Path) -> None:
    repo_root = tmp_path / "isaac-aura"
    artifacts_dir = repo_root / "artifacts" / "models"
    sibling_project = tmp_path / "isaac"
    artifacts_dir.mkdir(parents=True)
    sibling_project.mkdir()
    repo_model_path = artifacts_dir / DEFAULT_ULTRALYTICS_MODEL_NAME
    sibling_model_path = sibling_project / DEFAULT_ULTRALYTICS_MODEL_NAME
    repo_model_path.write_text("repo", encoding="utf-8")
    sibling_model_path.write_text("sibling", encoding="utf-8")

    assert default_model_path(repo_root=repo_root) == str(repo_model_path.resolve())
