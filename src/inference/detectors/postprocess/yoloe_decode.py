from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .segmentation_utils import clamp_bbox_xyxy, decode_mask_metadata, mask_centroid, mask_from_proto_coeff


@dataclass(frozen=True)
class DecodedYoloePrediction:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float] | None = None
    mask: np.ndarray | None = None
    mask_coeff: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_prediction_tensor(raw_predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(raw_predictions, dtype=np.float32)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError(f"YOLOE predictions must be rank-2 after squeeze, got {predictions.shape}")
    if predictions.shape[1] < 4 and predictions.shape[0] >= 4:
        return predictions.T.copy()
    if predictions.shape[0] >= 4 and predictions.shape[1] > predictions.shape[0] * 2:
        return predictions.T.copy()
    return predictions.copy()


def decode_yoloe_predictions(
    raw_predictions: np.ndarray,
    *,
    image_shape: tuple[int, int],
    num_classes: int,
    class_names: list[str] | None = None,
    confidence_threshold: float = 0.25,
    proto: np.ndarray | None = None,
    mask_dim: int = 0,
    mask_threshold: float = 0.5,
) -> list[DecodedYoloePrediction]:
    predictions = normalize_prediction_tensor(raw_predictions)
    names = class_names or [f"class_{idx}" for idx in range(num_classes)]
    attr_count = predictions.shape[1]
    if attr_count < 4 + num_classes:
        raise ValueError(
            "YOLOE prediction tensor is too small for bbox+class scores: "
            f"attrs={attr_count} num_classes={num_classes}"
        )
    bbox = predictions[:, :4]
    class_scores = predictions[:, 4 : 4 + num_classes]
    coeffs = None
    if mask_dim > 0:
        if attr_count < 4 + num_classes + mask_dim:
            raise ValueError(
                "YOLOE prediction tensor does not include requested mask coeffs: "
                f"attrs={attr_count} mask_dim={mask_dim}"
            )
        coeffs = predictions[:, 4 + num_classes : 4 + num_classes + mask_dim]

    height, width = int(image_shape[0]), int(image_shape[1])
    decoded: list[DecodedYoloePrediction] = []
    for index, row in enumerate(predictions):
        _ = row
        class_id = int(np.argmax(class_scores[index]))
        confidence = float(class_scores[index, class_id])
        if confidence < float(confidence_threshold):
            continue
        cx, cy, bw, bh = [float(value) for value in bbox[index]]
        x0 = (cx - bw * 0.5) * width
        y0 = (cy - bh * 0.5) * height
        x1 = (cx + bw * 0.5) * width
        y1 = (cy + bh * 0.5) * height
        bbox_xyxy = clamp_bbox_xyxy((x0, y0, x1, y1), width=width, height=height)

        mask = None
        centroid = ((bbox_xyxy[0] + bbox_xyxy[2]) * 0.5, (bbox_xyxy[1] + bbox_xyxy[3]) * 0.5)
        metadata: dict[str, Any] = {"decoder": "yoloe", "prediction_index": int(index)}
        coeff = None if coeffs is None else coeffs[index]
        if coeff is not None and proto is not None:
            mask = mask_from_proto_coeff(
                proto=np.asarray(proto, dtype=np.float32),
                coeff=np.asarray(coeff, dtype=np.float32),
                output_shape=(height, width),
                threshold=float(mask_threshold),
            )
            centroid = mask_centroid(mask) or centroid
            metadata.update(decode_mask_metadata(mask))

        decoded.append(
            DecodedYoloePrediction(
                class_id=class_id,
                class_name=names[class_id] if class_id < len(names) else f"class_{class_id}",
                confidence=confidence,
                bbox_xyxy=bbox_xyxy,
                centroid_xy=(float(centroid[0]), float(centroid[1])),
                mask=mask,
                mask_coeff=None if coeff is None else np.asarray(coeff, dtype=np.float32),
                metadata=metadata,
            )
        )
    return decoded
