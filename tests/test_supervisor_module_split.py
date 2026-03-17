from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime.supervisor import Supervisor


def test_process_frame_sequences_world_model_before_mission_update() -> None:
    supervisor = object.__new__(Supervisor)
    calls: list[object] = []
    enriched = object()

    def _update_world_model(batch):
        calls.append(("world", batch))
        return enriched

    def _update_mission(batch, *, publish=True):
        calls.append(("mission", batch, publish))
        return batch

    supervisor.update_world_model = _update_world_model  # type: ignore[method-assign]
    supervisor.update_mission = _update_mission  # type: ignore[method-assign]

    result = Supervisor.process_frame(supervisor, "frame-batch", publish=False)  # type: ignore[arg-type]

    assert result is enriched
    assert calls == [("world", "frame-batch"), ("mission", enriched, False)]
