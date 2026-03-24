from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.detectors.ultralytics_yolo import UltralyticsYoloDetector
from inference.detectors.ultralytics_yolo import load_tensorrt_engine_bytes


class _DummyBoxes:
    def __init__(self) -> None:
        self.xyxy = torch.tensor([[64.0, 64.0, 320.0, 320.0]], dtype=torch.float32)
        self.conf = torch.tensor([0.9], dtype=torch.float32)
        self.cls = torch.tensor([0.0], dtype=torch.float32)


class _DummyMasks:
    def __init__(self) -> None:
        self.data = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)


class _DummyResult:
    def __init__(self) -> None:
        self.boxes = _DummyBoxes()
        self.masks = _DummyMasks()
        self.names = {0: "chair"}


class _DummyDetectionResult:
    def __init__(self) -> None:
        self.boxes = _DummyBoxes()
        self.masks = None
        self.names = {0: "chair"}


class _DummyModel:
    def __init__(self, path: str) -> None:
        self.path = path
        self.names = {0: "chair"}
        self.last_source_shape: tuple[int, ...] | None = None
        self.last_conf: float | None = None

    def predict(self, *, source, conf, iou, max_det, device, verbose):  # noqa: ANN001
        _ = iou, max_det, device, verbose
        self.last_source_shape = tuple(int(v) for v in source.shape)
        self.last_conf = float(conf)
        return [_DummyResult()]


class _DummyDetectionModel(_DummyModel):
    def predict(self, *, source, conf, iou, max_det, device, verbose):  # noqa: ANN001
        _ = iou, max_det, device, verbose
        self.last_source_shape = tuple(int(v) for v in source.shape)
        self.last_conf = float(conf)
        return [_DummyDetectionResult()]


def test_ultralytics_detector_decodes_scaled_boxes_and_masks(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy.pt"
    model_path.write_text("stub", encoding="utf-8")
    detector = UltralyticsYoloDetector(
        str(model_path),
        imgsz=640,
        device="cpu",
        yolo_cls=_DummyModel,
    )

    rgb = np.zeros((96, 192, 3), dtype=np.uint8)
    results = detector.detect(rgb, timestamp=1.23, metadata={"target_class_hint": "chair"})

    assert detector.ready is True
    assert detector.model is not None
    assert detector.model.last_source_shape == (1, 3, 640, 640)
    assert detector.model.last_conf == 0.6
    assert len(results) == 1
    assert results[0].class_name == "chair"
    assert results[0].bbox_xyxy == (19, 10, 96, 48)
    assert results[0].mask is not None
    assert results[0].mask.shape == (96, 192)
    assert results[0].centroid_xy is not None
    assert results[0].metadata["backend"] == "ultralytics_yolo"
    assert results[0].metadata["raw_model_class_name"] == "chair"
    assert results[0].metadata["mask_area_px"] > 0


def test_ultralytics_detector_reports_missing_model(tmp_path: Path) -> None:
    detector = UltralyticsYoloDetector(str(tmp_path / "missing.pt"), device="cpu", yolo_cls=_DummyModel)

    assert detector.ready is False
    assert detector.probe().selected_reason == "model_missing"
    assert detector.info.warning != ""


def test_ultralytics_detector_decodes_boxes_without_segmentation_masks(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy.pt"
    model_path.write_text("stub", encoding="utf-8")
    detector = UltralyticsYoloDetector(
        str(model_path),
        imgsz=640,
        device="cpu",
        yolo_cls=_DummyDetectionModel,
    )

    rgb = np.zeros((96, 192, 3), dtype=np.uint8)
    results = detector.detect(rgb, timestamp=1.23, metadata={"target_class_hint": "chair"})

    assert detector.ready is True
    assert detector.model is not None
    assert detector.model.last_conf == 0.6
    assert len(results) == 1
    assert results[0].class_name == "chair"
    assert results[0].bbox_xyxy == (19, 10, 96, 48)
    assert results[0].mask is None
    assert results[0].centroid_xy == (57.5, 29.0)
    assert "mask_area_px" not in results[0].metadata
    assert results[0].metadata["raw_model_class_name"] == "chair"


class _AliasResult:
    def __init__(self) -> None:
        self.boxes = _DummyBoxes()
        self.masks = None
        self.names = {0: "television"}


class _UnsupportedResult:
    def __init__(self) -> None:
        self.boxes = _DummyBoxes()
        self.masks = None
        self.names = {0: "person"}


class _AliasModel(_DummyModel):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.names = {0: "television"}

    def predict(self, *, source, conf, iou, max_det, device, verbose):  # noqa: ANN001
        _ = iou, max_det, device, verbose
        self.last_source_shape = tuple(int(v) for v in source.shape)
        self.last_conf = float(conf)
        return [_AliasResult()]


class _UnsupportedModel(_DummyModel):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.names = {0: "person"}

    def predict(self, *, source, conf, iou, max_det, device, verbose):  # noqa: ANN001
        _ = iou, max_det, device, verbose
        self.last_source_shape = tuple(int(v) for v in source.shape)
        self.last_conf = float(conf)
        return [_UnsupportedResult()]


def test_ultralytics_detector_normalizes_aliases_to_supported_taxonomy(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy.pt"
    model_path.write_text("stub", encoding="utf-8")
    detector = UltralyticsYoloDetector(str(model_path), device="cpu", yolo_cls=_AliasModel)

    rgb = np.zeros((96, 192, 3), dtype=np.uint8)
    results = detector.detect(rgb, timestamp=1.23)

    assert len(results) == 1
    assert results[0].class_name == "tv"
    assert results[0].metadata["raw_model_class_name"] == "television"


def test_ultralytics_detector_skips_unsupported_classes(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy.pt"
    model_path.write_text("stub", encoding="utf-8")
    detector = UltralyticsYoloDetector(str(model_path), device="cpu", yolo_cls=_UnsupportedModel)

    rgb = np.zeros((96, 192, 3), dtype=np.uint8)
    results = detector.detect(rgb, timestamp=1.23)

    assert results == []
    assert "unsupported detector class skipped: person" in detector.probe().warnings


def test_load_tensorrt_engine_bytes_skips_ultralytics_metadata_header(tmp_path: Path) -> None:
    engine_path = tmp_path / "dummy.engine"
    metadata = b'{"description":"test-engine"}'
    payload = b"TRTENGINE"
    engine_path.write_bytes(len(metadata).to_bytes(4, byteorder="little") + metadata + payload)

    assert load_tensorrt_engine_bytes(engine_path) == payload
