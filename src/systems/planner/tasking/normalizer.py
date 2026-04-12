from __future__ import annotations

import re

from .ontology import ATTRIBUTES, OBJECT_CLASSES, ROOM_CLASSES

OBJECT_ALIASES = {
    "purple_box_cart": (
        "purple box cart",
        "purple box",
        "box cart",
        "storage box",
        "storage box cart",
    ),
    "tv": ("tv", "television"),
    "sofa": ("sofa", "couch"),
    "bed": ("bed",),
    "chair": ("chair",),
    "refrigerator": ("refrigerator", "fridge"),
    "door": ("door",),
}

ATTRIBUTE_ALIASES = {
    "power_state": (
        "power state",
        "power",
        "off",
        "on",
    ),
    "open_state": (
        "open state",
        "open",
        "closed",
    ),
}

ROOM_ALIASES = {
    "living_room": ("living room",),
    "bedroom": ("bedroom",),
    "kitchen": ("kitchen",),
}

FIND_KEYWORDS = ("find", "locate", "look for")
NAVIGATE_KEYWORDS = ("navigate", "go to", "move to", "approach", "head to")
INSPECT_KEYWORDS = ("inspect", "check", "status", "is ")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _find_alias_hits(text: str, alias_map: dict[str, tuple[str, ...]]) -> list[str]:
    normalized = normalize_text(text)
    hits: list[str] = []
    for canonical, aliases in alias_map.items():
        if any(alias in normalized for alias in aliases):
            hits.append(canonical)
    return hits


def detect_object_class(text: str) -> str | None:
    hits = _find_alias_hits(text, OBJECT_ALIASES)
    return hits[0] if hits else None


def detect_attribute(text: str, object_class: str | None = None) -> str | None:
    hits = _find_alias_hits(text, ATTRIBUTE_ALIASES)
    if object_class is None:
        return hits[0] if hits else None
    allowed = set(ATTRIBUTES.get(object_class, ()))
    for hit in hits:
        if hit in allowed:
            return hit
    if len(allowed) == 1 and any(keyword in normalize_text(text) for keyword in INSPECT_KEYWORDS):
        return next(iter(allowed))
    return None


def detect_room_hints(text: str, known_rooms: list[str] | None = None) -> list[str]:
    hits = _find_alias_hits(text, ROOM_ALIASES)
    if known_rooms is None:
        return hits
    allowed = set(known_rooms)
    return [room for room in hits if room in allowed]


def detect_instance_hint(text: str) -> str | None:
    normalized = normalize_text(text)
    if "left" in normalized:
        return "left"
    if "right" in normalized:
        return "right"
    match = re.search(r"(\d+)\s*(?:st|nd|rd|th)?", normalized)
    if match:
        return match.group(1)
    return None


def infer_intent(text: str, object_class: str | None, attribute: str | None) -> str:
    normalized = normalize_text(text)
    if object_class and any(token in normalized for token in FIND_KEYWORDS):
        return "find_object"
    if object_class and attribute is None and any(token in normalized for token in NAVIGATE_KEYWORDS):
        return "navigate_to_object"
    if object_class and attribute is not None:
        return "inspect_attribute"
    if object_class and any(token in normalized for token in INSPECT_KEYWORDS):
        return "inspect_attribute"
    return "unsupported"


def detect_desired_check(text: str, attribute: str | None) -> str:
    normalized = normalize_text(text)
    if attribute == "power_state":
        if "off" in normalized:
            return "is_off"
        if "on" in normalized:
            return "is_on"
        return "inspect"
    if attribute == "open_state":
        if "closed" in normalized:
            return "is_closed"
        if "open" in normalized:
            return "is_open"
        return "inspect"
    return "inspect"


def supports_object_name(value: str | None) -> bool:
    return value in OBJECT_CLASSES


def supports_room_name(value: str | None) -> bool:
    return value is None or value in ROOM_CLASSES
