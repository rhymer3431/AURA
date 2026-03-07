from __future__ import annotations

from ipc.messages import ActionCommand
from perception.speaker_events import SpeakerEvent


class SubgoalPlanner:
    def look_at(self, event: SpeakerEvent, *, task_id: str = "") -> ActionCommand:
        return ActionCommand(
            action_type="LOOK_AT",
            task_id=task_id,
            look_at_yaw_rad=float(event.direction_yaw_rad),
            metadata={"speaker_id": event.speaker_id, **event.metadata},
        )

    def follow_target(
        self,
        *,
        target_track_id: str,
        target_pose_xyz: tuple[float, float, float] | None,
        task_id: str = "",
    ) -> ActionCommand:
        return ActionCommand(
            action_type="FOLLOW_PERSON",
            task_id=task_id,
            target_track_id=target_track_id,
            target_pose_xyz=target_pose_xyz,
        )

    def nav_to_place(
        self,
        *,
        place_id: str,
        target_pose_xyz: tuple[float, float, float] | None,
        task_id: str = "",
        metadata: dict[str, object] | None = None,
    ) -> ActionCommand:
        return ActionCommand(
            action_type="NAV_TO_PLACE",
            task_id=task_id,
            target_place_id=place_id,
            target_pose_xyz=target_pose_xyz,
            metadata=dict(metadata or {}),
        )

    def nav_to_pose(
        self,
        *,
        target_pose_xyz: tuple[float, float, float],
        task_id: str = "",
        metadata: dict[str, object] | None = None,
    ) -> ActionCommand:
        return ActionCommand(
            action_type="NAV_TO_POSE",
            task_id=task_id,
            target_pose_xyz=target_pose_xyz,
            metadata=dict(metadata or {}),
        )

    def local_search(
        self,
        *,
        place_id: str = "",
        target_pose_xyz: tuple[float, float, float] | None = None,
        task_id: str = "",
        metadata: dict[str, object] | None = None,
    ) -> ActionCommand:
        return ActionCommand(
            action_type="LOCAL_SEARCH",
            task_id=task_id,
            target_place_id=place_id,
            target_pose_xyz=target_pose_xyz,
            metadata=dict(metadata or {}),
        )

    def stop(self, *, task_id: str = "", reason: str = "") -> ActionCommand:
        return ActionCommand(action_type="STOP", task_id=task_id, metadata={"reason": reason})
