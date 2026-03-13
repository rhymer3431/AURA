from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dashboard_backend.runtime_control import RuntimeControlClient


class _FakeHealth:
    @staticmethod
    def snapshot() -> dict[str, object]:
        return {"control": {"peer_count": 1}, "telemetry": {"peer_count": 0}}


class _FakeBus:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        self.health = _FakeHealth()

    def publish(self, topic: str, message) -> None:  # noqa: ANN001
        _ = topic, message

    def poll(self, topic: str, *, max_items: int | None = None):  # noqa: ANN001
        _ = topic, max_items
        return []

    def close(self) -> None:
        return None


def test_runtime_control_client_reads_health_property(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("dashboard_backend.runtime_control.ZmqBus", _FakeBus)

    client = RuntimeControlClient(
        control_endpoint="tcp://127.0.0.1:5580",
        telemetry_endpoint="tcp://127.0.0.1:5581",
    )

    try:
        assert client.transport_health_snapshot() == {
            "control": {"peer_count": 1},
            "telemetry": {"peer_count": 0},
        }
    finally:
        client.close()
