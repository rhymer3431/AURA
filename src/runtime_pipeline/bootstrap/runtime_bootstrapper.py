from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class RuntimeBootstrapper:
    startup_updates: int
    controller_setter: Callable[[object], None]
    ensure_runtime_bridge: Callable[[], None]
    ensure_control_server: Callable[[], None]
    initialize_server: Callable[[object, object], None]
    bootstrap_mode: Callable[[], None]
    seed_planner_overlay: Callable[[], None]
    publish_detector_capability: Callable[[], None]
    publish_ready_notice: Callable[[], None]

    def initialize(self, simulation_app, stage, controller) -> None:  # noqa: ANN001
        self.controller_setter(controller)
        for _ in range(max(int(self.startup_updates), 0)):
            simulation_app.update()
        self.ensure_runtime_bridge()
        self.ensure_control_server()
        self.initialize_server(simulation_app, stage)
        self.bootstrap_mode()
        self.seed_planner_overlay()
        self.publish_detector_capability()
        self.publish_ready_notice()

