"""Operator input controllers for cmd_vel and keyboard control."""

from __future__ import annotations

import threading
import time

import numpy as np


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


class KeyboardCmdVelController:
    """Receives GUI keyboard events and exposes SE(2) commands."""

    def __init__(self, lin_speed: float, lat_speed: float, yaw_speed: float, require_focus: bool = False):
        self.lin_speed = float(lin_speed)
        self.lat_speed = float(lat_speed)
        self.yaw_speed = float(yaw_speed)
        self.require_focus = bool(require_focus)
        self.quit_requested = False
        self.available = False

        self._pressed = {
            "W": False,
            "S": False,
            "Q": False,
            "E": False,
            "A": False,
            "D": False,
            "UP": False,
            "DOWN": False,
            "LEFT": False,
            "RIGHT": False,
        }
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._carb = None

        try:
            import carb
            import omni.appwindow
        except Exception as exc:
            if self.require_focus:
                raise RuntimeError(f"Keyboard input modules are unavailable: {exc}") from exc
            print(f"[WARN] Keyboard input modules are unavailable. Running with zero keyboard command: {exc}")
            return

        self._carb = carb
        try:
            self._input = carb.input.acquire_input_interface()
            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                raise RuntimeError("No app window is available. Keyboard control requires GUI mode.")
            self._keyboard = app_window.get_keyboard()
            if self._keyboard is None:
                raise RuntimeError("Keyboard handle is unavailable.")
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
            self.available = True
        except Exception as exc:
            self.shutdown()
            if self.require_focus:
                raise RuntimeError(f"Keyboard control is unavailable: {exc}") from exc
            print(f"[WARN] Keyboard control is unavailable. Running with zero keyboard command: {exc}")

    def print_help(self):
        print("[INFO] GUI keyboard control")
        print("[INFO]   W/S or UP/DOWN : forward / backward")
        print("[INFO]   Q/E            : strafe left / right")
        print("[INFO]   A/D or LEFT/RIGHT: yaw left / right")
        print("[INFO]   SPACE          : stop")
        print("[INFO]   ESC            : quit")
        if not self.available:
            print("[WARN] Keyboard events are not attached. Commands will stay at zero.")

    def _on_keyboard_event(self, event):
        key = event.input.name

        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            if key == "ESCAPE":
                self.quit_requested = True
                return True
            if key == "SPACE":
                for pressed_key in self._pressed:
                    self._pressed[pressed_key] = False
                return True
            if key in self._pressed:
                self._pressed[key] = True
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            if key in self._pressed:
                self._pressed[key] = False
        return True

    def _axis(self, positive_keys: tuple[str, ...], negative_keys: tuple[str, ...]) -> float:
        value = 0.0
        if any(self._pressed[key] for key in positive_keys):
            value += 1.0
        if any(self._pressed[key] for key in negative_keys):
            value -= 1.0
        return value

    def command(self) -> np.ndarray:
        vx = self.lin_speed * self._axis(("W", "UP"), ("S", "DOWN"))
        vy = self.lat_speed * self._axis(("Q",), ("E",))
        wz = self.yaw_speed * self._axis(("A", "LEFT"), ("D", "RIGHT"))
        return np.asarray((vx, vy, wz), dtype=np.float32)

    def shutdown(self):
        if self._input is not None and self._keyboard is not None and self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None
