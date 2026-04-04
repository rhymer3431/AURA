from __future__ import annotations

from typing import Literal, cast


ExecutionMode = Literal["TALK", "NAV", "MEM_NAV", "EXPLORE", "IDLE"]

EXECUTION_MODES: tuple[ExecutionMode, ...] = ("TALK", "NAV", "MEM_NAV", "EXPLORE", "IDLE")

_CANONICAL_MODES = set(EXECUTION_MODES)
_LEGACY_MODE_MAP = {
    "talk": "TALK",
    "nav": "NAV",
    "mem_nav": "MEM_NAV",
    "mem-nav": "MEM_NAV",
    "memnav": "MEM_NAV",
    "explore": "EXPLORE",
    "idle": "IDLE",
    "interactive": "NAV",
    "pointgoal": "MEM_NAV",
}


def normalize_execution_mode(value: object, *, default: ExecutionMode = "IDLE") -> ExecutionMode:
    raw = str(value or "").strip()
    if raw in _CANONICAL_MODES:
        return cast(ExecutionMode, raw)
    normalized = _LEGACY_MODE_MAP.get(raw.lower())
    if normalized is None:
        return default
    return cast(ExecutionMode, normalized)


def is_planning_mode(mode: object) -> bool:
    return normalize_execution_mode(mode) in {"NAV", "MEM_NAV", "EXPLORE"}


def uses_system2(mode: object) -> bool:
    return normalize_execution_mode(mode) == "NAV"
