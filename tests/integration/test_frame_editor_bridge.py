from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.inproc_bus import InprocBus
from runtime.frame_editor_bridge import AttachedFrameBridgeRuntime


class _FakeController:
    def __init__(self) -> None:
        self.forward_calls: list[tuple[int, tuple[float, float, float]]] = []

    def forward(self, frame_idx: int, command) -> None:  # noqa: ANN001
        vector = tuple(float(v) for v in np.asarray(command, dtype=np.float32).reshape(-1)[:3])
        self.forward_calls.append((int(frame_idx), vector))


class _FakeCommandSource:
    def __init__(self, args, *, bus, shm_ring=None) -> None:  # noqa: ANN001
        self.args = args
        self.bus = bus
        self.shm_ring = shm_ring
        self.initialized = False
        self.updated: list[int] = []

    def initialize(self, simulation_app, stage, controller) -> None:  # noqa: ANN001
        _ = simulation_app, stage, controller
        self.initialized = True

    def update(self, frame_idx: int) -> None:
        self.updated.append(int(frame_idx))

    def command(self) -> np.ndarray:
        return np.asarray([0.2, 0.0, 0.1], dtype=np.float32)

    def shutdown(self) -> None:
        self.initialized = False


def _args() -> Namespace:
    return Namespace(
        command="",
        startup_updates=0,
        strict_live=False,
        image_width=96,
        image_height=96,
        depth_max_m=5.0,
        cmd_max_vx=0.5,
        cmd_max_vy=0.3,
        cmd_max_wz=0.8,
        lookahead_distance_m=0.6,
        heading_slowdown_rad=0.6,
        traj_stale_timeout_sec=1.5,
        cmd_accel_limit=1.0,
        cmd_yaw_accel_limit=1.5,
        log_interval=1,
        physics_dt=1.0 / 60.0,
    )


def test_attached_frame_bridge_runtime_ticks_existing_controller() -> None:
    controller = _FakeController()
    runtime = AttachedFrameBridgeRuntime(
        args=_args(),
        controller=controller,
        bus=InprocBus(),
        command_source_factory=_FakeCommandSource,
    )

    runtime.start(simulation_app=object(), stage=object())
    command = runtime.tick(3)
    runtime.close()

    assert tuple(round(float(v), 4) for v in command) == (0.2, 0.0, 0.1)
    assert len(controller.forward_calls) == 1
    assert controller.forward_calls[0][0] == 3
    assert tuple(round(float(v), 4) for v in controller.forward_calls[0][1]) == (0.2, 0.0, 0.1)
