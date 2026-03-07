from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.temporal_store import TemporalMemoryStore


def test_temporal_memory_reacquire_returns_recent_follow_target_history() -> None:
    store = TemporalMemoryStore()
    store.record_event("person_track", timestamp=10.0, track_id="person_k", pose=(1.0, 0.0, 0.0))
    store.record_event("critic_flag", timestamp=11.0, track_id="person_k", payload={"reason": "occluded"})
    store.record_event("person_track", timestamp=12.0, track_id="person_k", pose=(1.5, 0.2, 0.0))
    store.record_event("person_track", timestamp=20.0, track_id="person_other", pose=(5.0, 0.0, 0.0))

    candidates = store.reacquire_track("person_k", now=13.0, max_age_sec=5.0)

    assert len(candidates) == 3
    assert candidates[0].track_id == "person_k"
    assert candidates[0].pose == (1.5, 0.2, 0.0)
