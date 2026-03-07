"""Console command input for cmd_vel control."""

from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np


class CommandSource(Protocol):
    """Produces SE(2) locomotion commands outside the physics callback."""

    quit_requested: bool

    def initialize(self, simulation_app, stage, controller) -> None:
        ...

    def update(self, frame_idx: int) -> None:
        ...

    def command(self) -> np.ndarray:
        ...

    def shutdown(self) -> None:
        ...


class ConsoleCmdVelController:
    """Receives cmd_vel commands from stdin and exposes SE(2) commands."""

    def __init__(self, timeout: float):
        self.timeout = max(0.0, float(timeout))
        self.quit_requested = False
        self._command = np.zeros(3, dtype=np.float32)
        self._last_command_time = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def print_help(self):
        print("[INFO] Console cmd_vel control")
        print("[INFO]   input format : vx vy wz")
        print("[INFO]   example      : 0.5 0.0 0.0")
        print("[INFO]   stop         : stop")
        print("[INFO]   quit         : quit")
        if self.timeout > 0.0:
            print(f"[INFO]   timeout      : {self.timeout:.2f}s -> zero command")
        else:
            print("[INFO]   timeout      : disabled")

    def initialize(self, simulation_app, stage, controller) -> None:
        del simulation_app, stage, controller

    def update(self, frame_idx: int) -> None:
        del frame_idx

    def _set_command(self, vx: float, vy: float, wz: float):
        with self._lock:
            self._command[:] = (vx, vy, wz)
            self._last_command_time = time.monotonic()

    def _input_loop(self):
        while self._running:
            try:
                raw = input("cmd_vel> ")
            except EOFError:
                return
            except Exception as exc:
                print(f"[WARN] cmd_vel input loop stopped: {exc}")
                return

            text = raw.strip()
            if not text:
                continue

            lowered = text.lower()
            if lowered in {"quit", "exit"}:
                self.quit_requested = True
                return
            if lowered in {"stop", "zero"}:
                self._set_command(0.0, 0.0, 0.0)
                continue
            if lowered == "help":
                self.print_help()
                continue

            parts = text.replace(",", " ").split()
            if len(parts) != 3:
                print("[WARN] Expected three values: vx vy wz")
                continue

            try:
                vx, vy, wz = (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError:
                print("[WARN] cmd_vel values must be numeric.")
                continue

            self._set_command(vx, vy, wz)

    def command(self) -> np.ndarray:
        with self._lock:
            command = self._command.copy()
            last_command_time = self._last_command_time

        if last_command_time == 0.0:
            return np.zeros(3, dtype=np.float32)
        if self.timeout > 0.0 and (time.monotonic() - last_command_time) > self.timeout:
            return np.zeros(3, dtype=np.float32)
        return command

    def shutdown(self):
        self._running = False
