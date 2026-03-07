from __future__ import annotations

from ipc.messages import ActionCommand
from memory.temporal_store import TemporalMemoryStore

from .subgoal_planner import SubgoalPlanner


class RecoveryPlanner:
    def __init__(self, temporal_store: TemporalMemoryStore, subgoal_planner: SubgoalPlanner) -> None:
        self._temporal = temporal_store
        self._subgoal_planner = subgoal_planner

    def recover_follow_target(self, track_id: str, *, now: float, task_id: str = "") -> ActionCommand | None:
        candidates = self._temporal.reacquire_track(track_id, now=now)
        for candidate in candidates:
            if candidate.pose is not None:
                return self._subgoal_planner.nav_to_pose(
                    target_pose_xyz=candidate.pose,
                    task_id=task_id,
                    metadata={"recovery": "temporal_reacquire", "track_id": track_id},
                )
        return None
