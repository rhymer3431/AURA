from __future__ import annotations

import socket
import sys
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.messages import TaskRequest
from ipc.shm_ring import SharedMemoryRing
from ipc.zmq_bus import ZmqBus


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_shared_memory_ring_roundtrip() -> None:
    name = f"isaac_aura_test_{uuid.uuid4().hex[:12]}"
    writer = SharedMemoryRing(name=name, slot_size=1024, capacity=4, create=True)
    reader = SharedMemoryRing(name=name, slot_size=1024, capacity=4, create=False)
    try:
        ref = writer.write(b"payload-123")
        assert reader.read(ref) == b"payload-123"
    finally:
        reader.close()
        writer.close(unlink=True)


def test_zmq_bus_roundtrip() -> None:
    pytest.importorskip("zmq")
    port = _free_tcp_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    server = ZmqBus(endpoint, bind=True)
    client = ZmqBus(endpoint, bind=False)
    try:
        client.publish("isaac.task", TaskRequest(command_text="follow"))
        deadline = time.time() + 2.0
        records = []
        while time.time() < deadline:
            records = server.poll("isaac.task", max_items=4)
            if records:
                break
            time.sleep(0.01)
        assert len(records) == 1
        assert records[0].message.command_text == "follow"
    finally:
        client.close()
        server.close()
