"""Runtime I/O facade for bridge initialization and publication hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RuntimeIOModule:
    """Thin publication wrapper used by NavigationRuntime phase 1."""

    open_bridge: Callable[[], Any]
    publish_detector_capability: Callable[[], None]
    publish_runtime_snapshot: Callable[..., None]

    def open(self):
        return self.open_bridge()

    def publish(self, *, frame_idx: int, update, evaluation) -> None:  # noqa: ANN001
        self.publish_runtime_snapshot(frame_idx, update=update, evaluation=evaluation)
