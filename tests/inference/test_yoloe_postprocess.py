from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.detectors.postprocess.yoloe_decode import decode_yoloe_predictions


def test_yoloe_bbox_decode_from_synthetic_tensor() -> None:
    raw = np.asarray(
        [
            [0.5, 0.5, 0.4, 0.4, 0.9, 0.1],
            [0.3, 0.3, 0.2, 0.2, 0.1, 0.8],
        ],
        dtype=np.float32,
    )

    decoded = decode_yoloe_predictions(raw, image_shape=(100, 200), num_classes=2, class_names=["apple", "person"])

    assert len(decoded) == 2
    assert decoded[0].class_name == "apple"
    assert decoded[0].bbox_xyxy == (60, 30, 140, 70)
    assert decoded[1].class_name == "person"


def test_yoloe_segmentation_decode_returns_mask_centroid() -> None:
    raw = np.asarray([[0.5, 0.5, 0.5, 0.5, 0.95, 5.0, 0.0]], dtype=np.float32)
    proto = np.asarray(
        [
            [[-5.0, -5.0], [-5.0, 5.0]],
            [[-5.0, -5.0], [-5.0, -5.0]],
        ],
        dtype=np.float32,
    )

    decoded = decode_yoloe_predictions(
        raw,
        image_shape=(4, 4),
        num_classes=1,
        class_names=["apple"],
        proto=proto,
        mask_dim=2,
        mask_threshold=0.5,
    )

    assert len(decoded) == 1
    assert decoded[0].mask is not None
    assert decoded[0].centroid_xy is not None
    assert decoded[0].centroid_xy[0] == 2.5
    assert decoded[0].centroid_xy[1] == 2.5
