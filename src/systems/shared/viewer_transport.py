"""Backward-compatible viewer-transport exports."""

from __future__ import annotations

from systems.shared.contracts.viewer_transport import (
    VIEWER_CONTROL_ENDPOINT,
    VIEWER_HEALTH_TOPIC,
    VIEWER_OBSERVATION_TOPIC,
    VIEWER_SHM_CAPACITY,
    VIEWER_SHM_NAME,
    VIEWER_SHM_SLOT_SIZE,
    VIEWER_TELEMETRY_ENDPOINT,
)

__all__ = [
    "VIEWER_CONTROL_ENDPOINT",
    "VIEWER_TELEMETRY_ENDPOINT",
    "VIEWER_SHM_NAME",
    "VIEWER_SHM_SLOT_SIZE",
    "VIEWER_SHM_CAPACITY",
    "VIEWER_OBSERVATION_TOPIC",
    "VIEWER_HEALTH_TOPIC",
]
