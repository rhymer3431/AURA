from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.models import ObsObject
from memory.spatial_store import SpatialMemoryStore


def test_spatial_memory_insert_update_and_association() -> None:
    store = SpatialMemoryStore()

    created = store.associate_observation(
        ObsObject(
            class_name="apple",
            track_id="apple-track-1",
            pose=(1.0, 2.0, 0.0),
            timestamp=10.0,
            confidence=0.8,
            movable=True,
        )
    )
    updated = store.associate_observation(
        ObsObject(
            class_name="apple",
            track_id="apple-track-1",
            pose=(1.2, 2.1, 0.0),
            timestamp=12.0,
            confidence=0.9,
            movable=True,
        )
    )

    assert created.matched_existing is False
    assert updated.matched_existing is True
    assert len(store.places) == 1
    assert len(store.objects) == 1
    assert updated.object_node.last_place_id == created.place_node.place_id
    assert updated.object_node.confidence == 0.9


def test_spatial_memory_flags_static_conflict_on_large_pose_change() -> None:
    store = SpatialMemoryStore(static_conflict_distance_m=1.0)

    store.associate_observation(
        ObsObject(
            class_name="table",
            track_id="table-1",
            pose=(0.0, 0.0, 0.0),
            timestamp=1.0,
            confidence=0.7,
            movable=False,
        )
    )
    updated = store.associate_observation(
        ObsObject(
            class_name="table",
            track_id="table-1",
            pose=(2.5, 0.0, 0.0),
            timestamp=2.0,
            confidence=0.8,
            movable=False,
        )
    )

    assert updated.conflict_flag is True
    assert updated.object_node.conflict_flag is True
