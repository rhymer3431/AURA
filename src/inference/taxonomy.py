from __future__ import annotations

import re

YOLO_CLASS_NAMES: tuple[str, ...] = (
    "bed",
    "book",
    "cabinet",
    "chair",
    "curtain",
    "door",
    "fridge",
    "lamp",
    "mirror",
    "pillow",
    "range_hood",
    "shelf",
    "sink",
    "sofa",
    "table",
    "toilet",
    "tv",
    "vase",
    "washing_machine",
    "window",
)

YOLO_CLASS_TO_ID: dict[str, int] = {name: index for index, name in enumerate(YOLO_CLASS_NAMES)}

_CLASS_NAME_NORMALIZER_RE = re.compile(r"[^a-z0-9]+")

_CLASS_NAME_ALIASES: dict[str, str] = {
    "basin": "sink",
    "books": "book",
    "bookshelf": "shelf",
    "cabinets": "cabinet",
    "ceiling_light": "lamp",
    "chandelier": "lamp",
    "closestool": "toilet",
    "couch": "sofa",
    "cupboard": "cabinet",
    "desk": "table",
    "dining_table": "table",
    "hood": "range_hood",
    "rangehood": "range_hood",
    "refrigerator": "fridge",
    "shelves": "shelf",
    "storage": "cabinet",
    "table_lamp": "lamp",
    "television": "tv",
    "throw_pillow": "pillow",
    "tvmonitor": "tv",
    "washer": "washing_machine",
    "washingmachine": "washing_machine",
}


def normalize_detector_class_name(class_name: str) -> str:
    normalized = _normalized_class_name(class_name)
    if normalized == "":
        return ""
    if normalized in YOLO_CLASS_TO_ID:
        return normalized
    return _CLASS_NAME_ALIASES.get(normalized, "")


def _normalized_class_name(class_name: str) -> str:
    raw = str(class_name).strip().lower()
    if raw == "":
        return ""
    return _CLASS_NAME_NORMALIZER_RE.sub("_", raw).strip("_")
