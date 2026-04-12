"""Shared-memory transport contract for runtime->navigation observations."""

from __future__ import annotations


NAVIGATION_SHM_NAME = "aura_navigation_observation_shm_01"
NAVIGATION_SHM_SLOT_SIZE = 8 * 1024 * 1024
NAVIGATION_SHM_CAPACITY = 8


__all__ = [
    "NAVIGATION_SHM_NAME",
    "NAVIGATION_SHM_SLOT_SIZE",
    "NAVIGATION_SHM_CAPACITY",
]
