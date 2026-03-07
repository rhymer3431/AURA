from __future__ import annotations

from ipc.messages import ActionCommand
from memory.temporal_store import TemporalMemoryStore

from .subgoal_planner import SubgoalPlanner


class RecoveryPlanner:
    def __init__(self, temporal_store: TemporalMemoryStore, subgoal_planner: SubgoalPlanner) -> None:
        self._temporal = temporal_store
        self._subgoal_planner = subgoal_planner

    def recover_follow_target(
        self,
        track_id: str,
        *,
        person_id: str = "",
        now: float,
        task_id: str = "",
        semantic_hints: list[dict[str, object]] | None = None,
    ) -> ActionCommand | None:
        if person_id != "":
            exact = self._temporal.reacquire_person(person_id, now=now)
            for candidate in exact:
                if candidate.pose is not None:
                    return self._subgoal_planner.nav_to_pose(
                        target_pose_xyz=candidate.pose,
                        task_id=task_id,
                        metadata={
                            "recovery": "recent_exact_person_match",
                            "person_id": person_id,
                            "track_id": track_id,
                            "semantic_hints": list(semantic_hints or []),
                        },
                    )
            reid_candidates = self._temporal.recent_events(
                event_type="reid_candidate",
                person_id=person_id,
                max_age_sec=6.0,
                now=now,
                limit=5,
            )
            for candidate in reid_candidates:
                if candidate.pose is not None:
                    return self._subgoal_planner.nav_to_pose(
                        target_pose_xyz=candidate.pose,
                        task_id=task_id,
                        metadata={
                            "recovery": "spatial_reid_candidate",
                            "person_id": person_id,
                            "track_id": track_id,
                            "semantic_hints": list(semantic_hints or []),
                        },
                    )
        last_visible = self._temporal.last_event(event_type="follow_loss", person_id=person_id, track_id=track_id)
        if last_visible is not None and last_visible.pose is not None:
            return self._subgoal_planner.nav_to_pose(
                target_pose_xyz=last_visible.pose,
                task_id=task_id,
                metadata={
                    "recovery": "last_visible_pose",
                    "person_id": person_id,
                    "track_id": track_id,
                    "semantic_hints": list(semantic_hints or []),
                },
            )
        candidates = self._temporal.reacquire_track(track_id, now=now)
        for candidate in candidates:
            if candidate.pose is not None:
                return self._subgoal_planner.nav_to_pose(
                    target_pose_xyz=candidate.pose,
                    task_id=task_id,
                    metadata={
                        "recovery": "temporal_reacquire",
                        "track_id": track_id,
                        "person_id": person_id,
                        "semantic_hints": list(semantic_hints or []),
                    },
                )
        return self._subgoal_planner.local_search(
            task_id=task_id,
            metadata={
                "recovery": "cone_search_local_search",
                "track_id": track_id,
                "person_id": person_id,
                "semantic_hints": list(semantic_hints or []),
            },
        )
