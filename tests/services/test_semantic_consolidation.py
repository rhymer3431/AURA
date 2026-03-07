from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.models import EpisodeRecord, ObsObject
from services.memory_service import MemoryService


def test_semantic_consolidation_updates_find_rule_from_episodes() -> None:
    memory_service = MemoryService()
    episode = EpisodeRecord(
        episode_id="episode_1",
        command_text="find apple",
        intent="goto_remembered_object",
        target_json={"target_class": "apple", "room_id": "kitchen"},
        candidate_object_ids=["object_1"],
        candidate_place_ids=["place_1"],
        success=True,
        started_at=1.0,
        ended_at=2.0,
    )
    memory_service.episodic_store.put(episode)
    memory_service.consolidator.consolidate_episode(episode.episode_id)

    failed_episode = EpisodeRecord(
        episode_id="episode_2",
        command_text="find apple again",
        intent="goto_remembered_object",
        target_json={"target_class": "apple", "room_id": "kitchen"},
        candidate_object_ids=["object_2"],
        candidate_place_ids=["place_2"],
        success=False,
        failure_reason="blocked",
        started_at=3.0,
        ended_at=4.0,
    )
    memory_service.episodic_store.put(failed_episode)
    memory_service.consolidator.consolidate_episode(failed_episode.episode_id)

    rules = memory_service.semantic_store.matching_rules(intent="find", target_class="apple", room_id="kitchen")
    assert len(rules) == 1
    assert rules[0].support_count == 2
    assert rules[0].rule_type == "object_search"
    assert rules[0].planner_hint["preferred_room"] == "kitchen"


def test_recall_object_returns_semantic_rules_and_applies_bonus() -> None:
    memory_service = MemoryService()
    kitchen_result = memory_service.observe_objects(
        [
            ObsObject(
                class_name="apple",
                track_id="apple_k",
                pose=(4.0, 0.0, 0.0),
                timestamp=10.0,
                confidence=0.85,
                room_id="kitchen",
            )
        ]
    )[0]
    memory_service.observe_objects(
        [
            ObsObject(
                class_name="apple",
                track_id="apple_h",
                pose=(1.0, 0.0, 0.0),
                timestamp=10.0,
                confidence=0.85,
                room_id="hall",
            )
        ]
    )
    memory_service.semantic_store.remember_rule(
        "find:apple:kitchen",
        "prefer kitchen recall",
        succeeded=True,
        trigger_signature="find:apple:kitchen",
        rule_type="object_search",
        planner_hint={
            "preferred_room": "kitchen",
            "preferred_place_id": kitchen_result.place_node.place_id,
            "preferred_object_id": kitchen_result.object_node.object_id,
        },
    )

    recall = memory_service.recall_object(
        query_text="아까 본 사과",
        target_class="apple",
        intent="find",
        current_pose=(0.0, 0.0, 0.0),
    )

    assert recall.semantic_rules
    assert recall.selected_object is not None
    assert recall.selected_object.object_id == kitchen_result.object_node.object_id
