from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.models import ObjectNode, ObjectSnapshot, PlaceNode, SemanticRule
from memory.working_memory import WorkingMemory


def test_working_memory_candidate_selection_prefers_recent_reachable_match() -> None:
    places = {
        "kitchen": PlaceNode(place_id="kitchen", pose=(1.0, 0.0, 0.0), room_id="kitchen", visit_count=2, first_seen=1.0, last_seen=10.0),
        "hall": PlaceNode(place_id="hall", pose=(8.0, 0.0, 0.0), room_id="hall", visit_count=1, first_seen=1.0, last_seen=10.0),
    }
    objects = [
        ObjectNode(
            object_id="apple_1",
            class_name="apple",
            track_id="a1",
            last_pose=(1.0, 0.2, 0.0),
            last_place_id="kitchen",
            first_seen=1.0,
            last_seen=10.0,
            confidence=0.9,
            snapshots=[ObjectSnapshot(timestamp=10.0, pose=(1.0, 0.2, 0.0), confidence=0.9)],
        ),
        ObjectNode(
            object_id="apple_2",
            class_name="apple",
            track_id="a2",
            last_pose=(8.0, 0.1, 0.0),
            last_place_id="hall",
            first_seen=1.0,
            last_seen=2.0,
            confidence=0.6,
            snapshots=[ObjectSnapshot(timestamp=2.0, pose=(8.0, 0.1, 0.0), confidence=0.6)],
        ),
    ]
    rules = [SemanticRule(rule_key="find:apple:kitchen", description="prefer kitchen tables", support_count=3, success_count=3, success_rate=1.0)]

    working_memory = WorkingMemory()
    candidates = working_memory.select_candidates(
        query_text="아까 봤던 사과를 찾아가",
        objects=objects,
        places=places,
        semantic_rules=rules,
        current_pose=(0.0, 0.0, 0.0),
        target_class="apple",
        room_id="kitchen",
        top_k=1,
    )

    assert candidates[0].object_id == "apple_1"
    assert candidates[0].active is True
    assert candidates[1].active is False
    assert candidates[0].score > candidates[1].score
