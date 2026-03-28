from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.models import ObsObject
from services.live_target_service import LiveTargetConfig, LiveTargetService


def _observation(
    *,
    class_name: str,
    track_id: str,
    pose: tuple[float, float, float],
    timestamp: float,
    depth_m: float,
    confidence: float = 0.9,
):
    return ObsObject(
        class_name=class_name,
        track_id=track_id,
        pose=pose,
        timestamp=timestamp,
        confidence=confidence,
        metadata={"depth_m": depth_m},
    )


def test_live_target_service_smooths_positions_and_targets_filtered_object_pose() -> None:
    service = LiveTargetService(LiveTargetConfig(ema_alpha_xy=0.5, default_object_standoff_m=0.9))
    service.activate_target(target_mode="goto_visible_object", target_class="apple")
    service.ingest_observations(
        [
            _observation(class_name="apple", track_id="apple_1", pose=(2.0, 0.0, 0.4), timestamp=1.0, depth_m=2.0),
            _observation(class_name="apple", track_id="apple_1", pose=(2.4, 0.0, 0.4), timestamp=1.1, depth_m=2.4),
        ]
    )

    snapshot = service.resolve_target(robot_pose_xyz=(0.0, 0.0, 0.0), now=1.2)

    assert snapshot is not None
    assert snapshot.track_id == "apple_1"
    assert tuple(round(float(value), 4) for value in snapshot.filtered_target_pose_xyz) == (2.2, 0.0, 0.4)
    assert tuple(round(float(value), 4) for value in snapshot.nav_goal_pose_xyz) == (2.2, 0.0, 0.4)
    assert snapshot.pose_source == "filtered_track"
    assert snapshot.command_metadata()["target_mode"] == "goto_visible_object"


def test_live_target_service_rejects_depth_jumps_and_targets_person_pose_inside_old_standoff() -> None:
    service = LiveTargetService(LiveTargetConfig(ema_alpha_xy=0.5, max_depth_jump_m=0.5, default_person_standoff_m=1.2))
    service.activate_target(target_mode="follow_person", target_track_id="person_1")
    service.ingest_observations(
        [
            _observation(class_name="person", track_id="person_1", pose=(0.8, 0.1, 0.0), timestamp=2.0, depth_m=0.81),
            _observation(class_name="person", track_id="person_1", pose=(3.0, 0.1, 0.0), timestamp=2.1, depth_m=3.0),
        ]
    )

    snapshot = service.resolve_target(robot_pose_xyz=(0.0, 0.0, 0.0), now=2.2)

    assert snapshot is not None
    assert tuple(round(float(value), 4) for value in snapshot.raw_target_pose_xyz) == (0.8, 0.1, 0.0)
    assert tuple(round(float(value), 4) for value in snapshot.nav_goal_pose_xyz) == (0.8, 0.1, 0.0)
    assert snapshot.visible is True


def test_live_target_service_returns_last_visible_pose_within_timeout_then_expires() -> None:
    service = LiveTargetService(LiveTargetConfig(default_loss_timeout_sec=1.0, visible_slack_sec=0.1))
    service.activate_target(target_mode="goto_visible_object", target_class="apple")
    service.ingest_observations(
        [_observation(class_name="apple", track_id="apple_1", pose=(3.0, 0.0, 0.5), timestamp=5.0, depth_m=3.0)]
    )

    last_visible = service.resolve_target(robot_pose_xyz=(0.0, 0.0, 0.0), now=5.6)
    stale_only = service.resolve_target(robot_pose_xyz=(0.0, 0.0, 0.0), now=6.5, allow_stale=True)
    expired = service.resolve_target(robot_pose_xyz=(0.0, 0.0, 0.0), now=6.5)

    assert last_visible is not None
    assert last_visible.pose_source == "last_visible"
    assert last_visible.within_loss_timeout is True
    assert stale_only is not None
    assert stale_only.within_loss_timeout is False
    assert expired is None
