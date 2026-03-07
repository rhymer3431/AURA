from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.models import ObsObject
from services.memory_service import MemoryService


def test_object_recall_query_returns_object_and_anchor_place() -> None:
    memory_service = MemoryService()
    memory_service.observe_objects(
        [
            ObsObject(
                class_name="apple",
                track_id="apple-track-1",
                pose=(2.0, -1.0, 0.0),
                timestamp=100.0,
                confidence=0.95,
                room_id="kitchen",
            )
        ]
    )
    memory_service.semantic_store.remember_rule(
        "find:apple:kitchen",
        "prioritize table-like surfaces",
        succeeded=True,
    )

    recall = memory_service.recall_object(
        query_text="아까 봤던 사과를 찾아가",
        target_class="apple",
        intent="find",
        room_id="kitchen",
        current_pose=(0.0, 0.0, 0.0),
    )

    assert recall.selected_object is not None
    assert recall.selected_place is not None
    assert recall.selected_object.class_name == "apple"
    assert recall.selected_place.place_id == recall.selected_object.last_place_id
