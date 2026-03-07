from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SocketHealth:
    plane: str
    last_send_ns: int = 0
    last_recv_ns: int = 0
    reconnect_attempts: int = 0
    queued_messages: int = 0
    dropped_messages: int = 0
    last_error: str = ""

    def on_send(self) -> None:
        self.last_send_ns = time.time_ns()

    def on_recv(self) -> None:
        self.last_recv_ns = time.time_ns()

    def on_retry(self) -> None:
        self.reconnect_attempts += 1

    def on_queue(self, count: int) -> None:
        self.queued_messages = int(count)

    def on_drop(self, count: int = 1, *, error: str = "") -> None:
        self.dropped_messages += int(count)
        if error != "":
            self.last_error = str(error)

    def as_dict(self) -> dict[str, object]:
        return {
            "plane": self.plane,
            "last_send_ns": self.last_send_ns,
            "last_recv_ns": self.last_recv_ns,
            "reconnect_attempts": self.reconnect_attempts,
            "queued_messages": self.queued_messages,
            "dropped_messages": self.dropped_messages,
            "last_error": self.last_error,
        }


@dataclass
class TransportHealthTracker:
    control: SocketHealth = field(default_factory=lambda: SocketHealth(plane="control"))
    telemetry: SocketHealth = field(default_factory=lambda: SocketHealth(plane="telemetry"))

    def snapshot(self) -> dict[str, object]:
        return {
            "control": self.control.as_dict(),
            "telemetry": self.telemetry.as_dict(),
        }
