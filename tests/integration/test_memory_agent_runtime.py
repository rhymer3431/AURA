from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.inproc_bus import InprocBus
from runtime.memory_agent_runtime import MemoryAgentRuntime


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        memory_db_path=str(tmp_path / "memory.sqlite"),
        bootstrap_rule="find:apple:kitchen",
        bus="inproc",
        endpoint="tcp://127.0.0.1:5560",
        bind=False,
        control_endpoint="",
        telemetry_endpoint="",
        shm_name="isaac_aura_frames_test",
        shm_slot_size=1024 * 1024,
        shm_capacity=2,
        command="아까 봤던 사과를 찾아가",
        scene="apple",
        loopback=True,
        detector_model_path="artifacts/models/__missing__.engine",
        frame_source="synthetic",
        strict_live=False,
        serve=True,
        once=False,
        poll_interval_ms=1,
        health_interval_sec=0.01,
        persist_interval_sec=0.01,
        max_cycles=2,
        idle_exit_after_sec=0.0,
    )


def test_memory_agent_runtime_runs_persistent_loop_and_persists_snapshot(tmp_path: Path) -> None:
    runtime = MemoryAgentRuntime(_args(tmp_path), bus=InprocBus(), shm_ring=None)
    try:
        assert runtime.run() == 0
    finally:
        runtime.close()

    assert (tmp_path / "memory.sqlite").exists()
