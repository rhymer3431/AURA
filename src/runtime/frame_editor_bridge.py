from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipc.base import MessageBus
from ipc.shm_ring import SharedMemoryRing

from .frame_bridge_runtime import FrameBridgeCommandSource


class _NoopSimulationApp:
    def update(self) -> None:
        return None


def editor_bridge_available() -> tuple[bool, str]:
    try:
        import omni.usd  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"Omniverse USD context unavailable: {type(exc).__name__}: {exc}"
    return True, ""


@dataclass
class AttachedFrameBridgeRuntime:
    args: object
    controller: object
    bus: MessageBus
    shm_ring: SharedMemoryRing | None = None
    command_source_factory: object = FrameBridgeCommandSource

    def __post_init__(self) -> None:
        self._command_source = None
        self._frame_idx = 0
        self._started = False

    @property
    def command_source(self):
        return self._command_source

    def start(self, *, simulation_app=None, stage=None) -> None:
        resolved_stage = stage if stage is not None else self._resolve_stage()
        resolved_app = simulation_app if simulation_app is not None else self._resolve_simulation_app()
        self._command_source = self.command_source_factory(self.args, bus=self.bus, shm_ring=self.shm_ring)
        self._command_source.initialize(resolved_app, resolved_stage, self.controller)
        self._started = True

    def tick(self, frame_idx: int | None = None) -> np.ndarray:
        if not self._started or self._command_source is None:
            raise RuntimeError("AttachedFrameBridgeRuntime.start() must be called before tick().")
        current_idx = self._frame_idx + 1 if frame_idx is None else int(frame_idx)
        self._frame_idx = current_idx
        self._command_source.update(current_idx)
        command = self._command_source.command()
        forward = getattr(self.controller, "forward", None)
        if callable(forward):
            forward(current_idx, command)
        return command

    def close(self) -> None:
        if self._command_source is not None:
            self._command_source.shutdown()
        self._started = False

    @staticmethod
    def _resolve_stage():
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage found for editor bridge attach.")
        return stage

    @staticmethod
    def _resolve_simulation_app():
        try:
            import omni.kit.app
        except Exception:  # noqa: BLE001
            return _NoopSimulationApp()
        app = omni.kit.app.get_app()
        if app is None:
            return _NoopSimulationApp()
        return app
