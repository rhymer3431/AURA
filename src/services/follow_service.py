from __future__ import annotations

from control.subgoal_planner import SubgoalPlanner
from memory.temporal_store import TemporalMemoryStore
from perception.person_tracker import PersonTracker


class FollowService:
    def __init__(self, temporal_store: TemporalMemoryStore, person_tracker: PersonTracker, subgoal_planner: SubgoalPlanner) -> None:
        self._temporal = temporal_store
        self._tracker = person_tracker
        self._subgoal_planner = subgoal_planner
        self._target_track_id = ""

    @property
    def target_track_id(self) -> str:
        return self._target_track_id

    def bind_target(self, track_id: str) -> None:
        self._target_track_id = str(track_id)

    def ingest_observations(self, observations) -> None:  # noqa: ANN001
        self._tracker.update(list(observations))
        for track in self._tracker.all_tracks():
            self._temporal.record_event(
                "person_track",
                timestamp=float(track.last_seen),
                track_id=track.track_id,
                pose=track.last_pose,
                payload={"confidence": float(track.confidence)},
            )
        if self._target_track_id == "":
            latest = self._tracker.all_tracks()
            if latest:
                self._target_track_id = latest[0].track_id

    def build_command(self, *, task_id: str = ""):
        if self._target_track_id == "":
            return None
        track = self._tracker.get(self._target_track_id)
        if track is None:
            return None
        return self._subgoal_planner.follow_target(
            target_track_id=self._target_track_id,
            target_pose_xyz=track.last_pose,
            task_id=task_id,
        )
