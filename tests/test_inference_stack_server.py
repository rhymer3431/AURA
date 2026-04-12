from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
import time

from systems.inference.stack import server as inference_stack_server


def test_inference_service_snapshot_probes_services_in_parallel(monkeypatch) -> None:
    server = inference_stack_server.InferenceSystemServer.__new__(inference_stack_server.InferenceSystemServer)
    server.args = Namespace(health_timeout=1.0)
    server._services = [
        SimpleNamespace(name="navdp", base_url="http://navdp", health_url="http://navdp/healthz"),
        SimpleNamespace(name="system2", base_url="http://system2", health_url="http://system2/healthz"),
        SimpleNamespace(name="planner", base_url="http://planner", health_url="http://planner/healthz"),
    ]

    def fake_json_get(url: str, *, timeout_s: float):
        del timeout_s
        time.sleep(0.2)
        return "healthy", {"url": url}, 200.0, None

    monkeypatch.setattr(inference_stack_server, "_json_get", fake_json_get)

    started = time.perf_counter()
    snapshot = server._service_snapshot()
    elapsed = time.perf_counter() - started

    assert set(snapshot.keys()) == {"navdp", "system2", "planner"}
    assert elapsed < 0.45


def test_inference_models_state_only_requires_required_services(monkeypatch) -> None:
    server = inference_stack_server.InferenceSystemServer.__new__(inference_stack_server.InferenceSystemServer)
    server.args = Namespace(host="127.0.0.1", port=15880, log_dir="logs")
    server._services = [
        SimpleNamespace(name="navdp", base_url="http://navdp", health_url="http://navdp/healthz", required=False),
        SimpleNamespace(name="system2", base_url="http://system2", health_url="http://system2/healthz", required=True),
        SimpleNamespace(name="planner", base_url="http://planner", health_url="http://planner/health", required=False),
    ]
    server._registry = SimpleNamespace(snapshot=lambda: [])

    monkeypatch.setattr(
        server,
        "_service_snapshot",
        lambda service_name=None: {
            "navdp": {"status": "unreachable"},
            "system2": {"status": "healthy"},
            "planner": {"status": "error"},
        },
    )

    state = server.models_state()

    assert state["ok"] is True
    assert state["models"]["system2"]["status"] == "healthy"
    assert state["models"]["navdp"]["status"] == "unreachable"
