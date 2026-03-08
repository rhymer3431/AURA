from __future__ import annotations

from control.subgoal_planner import SubgoalPlanner
from memory.semantic_store import SemanticMemoryStore
from memory.temporal_store import TemporalMemoryStore
from perception.person_tracker import PersonTracker, PersonTrack

from .live_target_service import LiveTargetService


class FollowService:
    def __init__(
        self,
        temporal_store: TemporalMemoryStore,
        person_tracker: PersonTracker,
        subgoal_planner: SubgoalPlanner,
        live_target_service: LiveTargetService,
        semantic_store: SemanticMemoryStore | None = None,
    ) -> None:
        self._temporal = temporal_store
        self._tracker = person_tracker
        self._subgoal_planner = subgoal_planner
        self._live_target = live_target_service
        self._semantic = semantic_store
        self._target_track_id = ""
        self._target_person_id = ""

    @property
    def target_track_id(self) -> str:
        return self._target_track_id

    @property
    def target_person_id(self) -> str:
        return self._target_person_id

    def bind_target(self, track_id: str = "", *, person_id: str = "") -> None:
        self._target_track_id = str(track_id)
        self._target_person_id = str(person_id)
        if self._target_person_id == "" and self._target_track_id != "":
            self._target_person_id = self._tracker.person_id_for_track(self._target_track_id)
        if self._target_track_id != "":
            self._live_target.bind_active_track(self._target_track_id)

    def clear_target(self) -> None:
        self._target_track_id = ""
        self._target_person_id = ""

    def ingest_tracks(self, tracks: list[PersonTrack], *, now: float) -> None:
        for track in tracks:
            self._temporal.record_event(
                "person_track",
                timestamp=float(track.last_seen),
                track_id=track.track_id,
                person_id=track.person_id,
                pose=track.last_pose,
                payload={
                    "confidence": float(track.confidence),
                    "appearance_signature": track.appearance_signature,
                    **track.score_breakdown,
                },
            )
        if self._target_person_id == "" and tracks:
            self.bind_target(tracks[0].track_id, person_id=tracks[0].person_id)
            return
        if self._target_person_id == "":
            return
        for track in tracks:
            if track.person_id == self._target_person_id:
                self._target_track_id = track.track_id
                self._temporal.record_event(
                    "reid_candidate",
                    timestamp=float(now),
                    track_id=track.track_id,
                    person_id=track.person_id,
                    pose=track.last_pose,
                    payload=dict(track.score_breakdown),
                )

    def build_command(
        self,
        *,
        task_id: str = "",
        robot_pose_xyz: tuple[float, float, float] | None = None,
        now: float | None = None,
    ):
        track = self._resolve_target_track()
        if track is None:
            return None
        semantic_hints = self.recovery_semantic_hints()
        target_pose_xyz = track.last_pose
        metadata: dict[str, object] = {
            "semantic_hints": semantic_hints,
            "target_mode": "follow_person",
            "pose_source": "person_track_fallback",
            "raw_target_pose_xyz": list(track.last_pose),
            "nav_goal_pose_xyz": list(track.last_pose),
            "track_age_sec": 0.0 if now is None else max(float(now) - float(track.last_seen), 0.0),
        }
        if robot_pose_xyz is not None and now is not None:
            snapshot = self._live_target.resolve_target(
                robot_pose_xyz=robot_pose_xyz,
                now=float(now),
                target_mode="follow_person",
                target_class="person",
                target_track_id=track.track_id,
                preferred_track_id=track.track_id,
            )
            if snapshot is not None:
                target_pose_xyz = snapshot.nav_goal_pose_xyz
                metadata.update(snapshot.command_metadata())
        return self._subgoal_planner.follow_target(
            target_track_id=track.track_id,
            target_person_id=track.person_id,
            target_pose_xyz=target_pose_xyz,
            task_id=task_id,
            metadata=metadata,
        )

    def mark_target_lost(self, *, now: float) -> None:
        track = self._resolve_target_track()
        self._temporal.record_event(
            "follow_loss",
            timestamp=float(now),
            track_id=self._target_track_id,
            person_id=self._target_person_id,
            pose=None if track is None else track.last_pose,
            payload={"semantic_hints": self.recovery_semantic_hints()},
        )

    def recovery_semantic_hints(self) -> list[dict[str, object]]:
        if self._semantic is None:
            return []
        rules = self._semantic.matching_rules(intent="follow", target_class="person", room_id="")
        hints: list[dict[str, object]] = []
        for rule in rules[:3]:
            hints.append(
                {
                    "rule_key": rule.rule_key,
                    "planner_hint": dict(rule.planner_hint),
                    "success_rate": float(rule.success_rate),
                }
            )
        return hints

    def _resolve_target_track(self) -> PersonTrack | None:
        if self._target_person_id != "":
            track = self._tracker.get_by_person_id(self._target_person_id)
            if track is not None:
                self._target_track_id = track.track_id
                return track
        if self._target_track_id != "":
            track = self._tracker.get(self._target_track_id)
            if track is not None:
                self._target_person_id = track.person_id
                self._target_track_id = track.track_id
                return track
        return None
