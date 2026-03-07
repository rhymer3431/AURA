from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perception.reid_store import ReIdStore


def test_reid_prefers_embedding_and_continuity_over_false_near_candidate() -> None:
    store = ReIdStore()
    identity_a, _ = store.assign(
        track_id="track_a",
        pose=(0.6, 0.0, 0.0),
        timestamp=1.0,
        confidence=0.9,
        appearance_signature="blue",
    )
    store.assign(
        track_id="track_b",
        pose=(0.05, 0.0, 0.0),
        timestamp=1.0,
        confidence=0.9,
        appearance_signature="red",
    )

    matches = store.candidate_matches(
        pose=(0.1, 0.0, 0.0),
        timestamp=1.2,
        appearance_signature="blue",
    )

    assert matches[0].person_id == identity_a.person_id
    assert matches[0].embedding_match > 0.0
