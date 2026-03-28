from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from memory.models import ObsObject, pose3


_VISIBLE_TARGET_MODES = {"follow_person", "goto_visible_object"}


@dataclass(frozen=True)
class LiveTargetConfig:
    ema_alpha_xy: float = 0.45
    max_depth_jump_m: float = 1.25
    visible_slack_sec: float = 0.35
    default_loss_timeout_sec: float = 1.5
    default_person_standoff_m: float = 1.2
    default_object_standoff_m: float = 0.9


@dataclass(frozen=True)
class LiveTargetRequest:
    target_mode: str = ""
    target_class: str = ""
    target_track_id: str = ""
    active_track_id: str = ""
    standoff_distance_m: float | None = None
    loss_timeout_sec: float | None = None


@dataclass(frozen=True)
class LiveTargetSnapshot:
    target_mode: str
    target_class: str
    track_id: str
    raw_target_pose_xyz: tuple[float, float, float]
    filtered_target_pose_xyz: tuple[float, float, float]
    nav_goal_pose_xyz: tuple[float, float, float]
    approach_yaw_rad: float
    depth_m: float
    track_age_sec: float
    pose_source: str
    visible: bool
    within_loss_timeout: bool

    def command_metadata(self) -> dict[str, object]:
        return {
            "target_mode": self.target_mode,
            "target_class": self.target_class,
            "target_track_id": self.track_id,
            "pose_source": self.pose_source,
            "raw_target_pose_xyz": list(self.raw_target_pose_xyz),
            "filtered_target_pose_xyz": list(self.filtered_target_pose_xyz),
            "nav_goal_pose_xyz": list(self.nav_goal_pose_xyz),
            "approach_yaw_rad": float(self.approach_yaw_rad),
            "track_age_sec": float(self.track_age_sec),
            "depth_m": float(self.depth_m),
            "target_visible": bool(self.visible),
            "target_within_loss_timeout": bool(self.within_loss_timeout),
        }


@dataclass
class _TrackState:
    track_id: str
    class_name: str
    raw_pose_xyz: tuple[float, float, float]
    filtered_xy: np.ndarray
    last_seen: float
    confidence: float
    depth_m: float
    last_reject_reason: str = ""

    @property
    def filtered_pose_xyz(self) -> tuple[float, float, float]:
        return (
            float(self.filtered_xy[0]),
            float(self.filtered_xy[1]),
            float(self.raw_pose_xyz[2]),
        )


class LiveTargetService:
    def __init__(self, config: LiveTargetConfig | None = None) -> None:
        self.config = config or LiveTargetConfig()
        self._request = LiveTargetRequest()
        self._tracks: dict[str, _TrackState] = {}

    @property
    def request(self) -> LiveTargetRequest:
        return self._request

    def activate_target(
        self,
        *,
        target_mode: str,
        target_class: str = "",
        target_track_id: str = "",
        standoff_distance_m: float | None = None,
        loss_timeout_sec: float | None = None,
    ) -> None:
        normalized_mode = str(target_mode).strip().lower()
        if normalized_mode not in _VISIBLE_TARGET_MODES:
            self._request = LiveTargetRequest()
            return
        normalized_class = str(target_class).strip().lower()
        if normalized_mode == "follow_person":
            normalized_class = "person"
        normalized_track_id = str(target_track_id).strip()
        self._request = LiveTargetRequest(
            target_mode=normalized_mode,
            target_class=normalized_class,
            target_track_id=normalized_track_id,
            active_track_id=normalized_track_id,
            standoff_distance_m=self._normalize_positive(standoff_distance_m),
            loss_timeout_sec=self._normalize_positive(loss_timeout_sec),
        )

    def clear_target(self) -> None:
        self._request = LiveTargetRequest()

    def bind_active_track(self, track_id: str) -> None:
        normalized_track_id = str(track_id).strip()
        if normalized_track_id == "" or self._request.target_mode == "":
            return
        self._request = LiveTargetRequest(
            target_mode=self._request.target_mode,
            target_class=self._request.target_class,
            target_track_id=self._request.target_track_id,
            active_track_id=normalized_track_id,
            standoff_distance_m=self._request.standoff_distance_m,
            loss_timeout_sec=self._request.loss_timeout_sec,
        )

    def ingest_observations(self, observations: list[ObsObject]) -> None:
        alpha = float(np.clip(self.config.ema_alpha_xy, 0.0, 1.0))
        for observation in observations:
            track_id = str(observation.track_id).strip()
            if track_id == "":
                continue
            pose = np.asarray(pose3(observation.pose), dtype=np.float32)
            if pose.shape[0] < 3 or not np.all(np.isfinite(pose[:3])):
                continue
            depth_m = self._resolve_depth_m(observation)
            if depth_m <= 0.0 or not np.isfinite(depth_m):
                continue

            previous = self._tracks.get(track_id)
            if previous is not None and abs(float(depth_m) - float(previous.depth_m)) > float(self.config.max_depth_jump_m):
                previous.last_reject_reason = "depth_jump"
                continue

            if previous is None:
                filtered_xy = pose[:2].copy()
            else:
                filtered_xy = (1.0 - alpha) * previous.filtered_xy + alpha * pose[:2]
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                class_name=str(observation.class_name).strip().lower(),
                raw_pose_xyz=(float(pose[0]), float(pose[1]), float(pose[2])),
                filtered_xy=np.asarray(filtered_xy, dtype=np.float32),
                last_seen=float(observation.timestamp),
                confidence=float(observation.confidence),
                depth_m=float(depth_m),
                last_reject_reason="",
            )

    def resolve_target(
        self,
        *,
        robot_pose_xyz: tuple[float, float, float] | np.ndarray,
        now: float,
        target_mode: str = "",
        target_class: str = "",
        target_track_id: str = "",
        preferred_track_id: str = "",
        standoff_distance_m: float | None = None,
        loss_timeout_sec: float | None = None,
        allow_stale: bool = False,
    ) -> LiveTargetSnapshot | None:
        mode = str(target_mode).strip().lower() or self._request.target_mode
        if mode not in _VISIBLE_TARGET_MODES:
            return None
        requested_class = str(target_class).strip().lower() or self._request.target_class
        if mode == "follow_person":
            requested_class = "person"
        preferred = str(preferred_track_id).strip()
        explicit = str(target_track_id).strip()
        state = self._select_track_state(
            preferred_track_id=preferred,
            explicit_track_id=explicit,
            requested_class=requested_class,
        )
        if state is None:
            return None

        active_track_id = state.track_id
        self._request = LiveTargetRequest(
            target_mode=mode,
            target_class=requested_class,
            target_track_id=str(self._request.target_track_id or explicit),
            active_track_id=active_track_id,
            standoff_distance_m=self._request.standoff_distance_m
            if standoff_distance_m is None
            else self._normalize_positive(standoff_distance_m),
            loss_timeout_sec=self._request.loss_timeout_sec
            if loss_timeout_sec is None
            else self._normalize_positive(loss_timeout_sec),
        )

        age_sec = max(0.0, float(now) - float(state.last_seen))
        timeout_sec = self._loss_timeout_for(mode=mode, override=loss_timeout_sec)
        if age_sec > timeout_sec and not allow_stale:
            return None

        robot_pose = np.asarray(pose3(robot_pose_xyz), dtype=np.float32)
        filtered_pose = np.asarray(state.filtered_pose_xyz, dtype=np.float32)
        raw_pose = np.asarray(state.raw_pose_xyz, dtype=np.float32)
        standoff_m = self._standoff_distance_for(mode=mode, override=standoff_distance_m)
        nav_goal_xyz = self._compute_nav_goal_xyz(
            target_pose_xyz=filtered_pose,
            robot_pose_xyz=robot_pose,
            standoff_distance_m=standoff_m,
        )
        delta_xy = filtered_pose[:2] - robot_pose[:2]
        if float(np.linalg.norm(delta_xy)) <= 1.0e-6:
            approach_yaw_rad = 0.0
        else:
            approach_yaw_rad = float(math.atan2(float(delta_xy[1]), float(delta_xy[0])))
        visible = age_sec <= float(self.config.visible_slack_sec)
        return LiveTargetSnapshot(
            target_mode=mode,
            target_class=requested_class or state.class_name,
            track_id=state.track_id,
            raw_target_pose_xyz=(float(raw_pose[0]), float(raw_pose[1]), float(raw_pose[2])),
            filtered_target_pose_xyz=(float(filtered_pose[0]), float(filtered_pose[1]), float(filtered_pose[2])),
            nav_goal_pose_xyz=(float(nav_goal_xyz[0]), float(nav_goal_xyz[1]), float(nav_goal_xyz[2])),
            approach_yaw_rad=float(approach_yaw_rad),
            depth_m=float(state.depth_m),
            track_age_sec=float(age_sec),
            pose_source="filtered_track" if visible else "last_visible",
            visible=bool(visible),
            within_loss_timeout=bool(age_sec <= timeout_sec),
        )

    def _select_track_state(
        self,
        *,
        preferred_track_id: str,
        explicit_track_id: str,
        requested_class: str,
    ) -> _TrackState | None:
        for candidate_track_id in (
            str(preferred_track_id).strip(),
            str(explicit_track_id).strip(),
            str(self._request.target_track_id).strip(),
            str(self._request.active_track_id).strip(),
        ):
            if candidate_track_id == "":
                continue
            state = self._tracks.get(candidate_track_id)
            if state is not None and (requested_class == "" or state.class_name == requested_class):
                return state

        candidates = [
            state
            for state in self._tracks.values()
            if requested_class == "" or state.class_name == requested_class
        ]
        if not candidates:
            return None

        anchor = self._tracks.get(str(self._request.active_track_id).strip())
        if anchor is not None:
            candidates.sort(
                key=lambda item: (
                    float(np.linalg.norm(item.filtered_xy - anchor.filtered_xy)),
                    -float(item.last_seen),
                    -float(item.confidence),
                )
            )
            return candidates[0]

        candidates.sort(key=lambda item: (-float(item.last_seen), -float(item.confidence)))
        return candidates[0]

    def _standoff_distance_for(self, *, mode: str, override: float | None) -> float:
        normalized_override = self._normalize_positive(override)
        if normalized_override is None:
            normalized_override = self._request.standoff_distance_m
        if normalized_override is not None:
            return float(normalized_override)
        if str(mode).strip().lower() == "follow_person":
            return float(self.config.default_person_standoff_m)
        return float(self.config.default_object_standoff_m)

    def _loss_timeout_for(self, *, mode: str, override: float | None) -> float:
        normalized_override = self._normalize_positive(override)
        if normalized_override is None:
            normalized_override = self._request.loss_timeout_sec
        if normalized_override is not None:
            return float(normalized_override)
        _ = mode
        return float(self.config.default_loss_timeout_sec)

    @staticmethod
    def _compute_nav_goal_xyz(
        *,
        target_pose_xyz: np.ndarray,
        robot_pose_xyz: np.ndarray,
        standoff_distance_m: float,
    ) -> np.ndarray:
        del robot_pose_xyz, standoff_distance_m
        return np.asarray(target_pose_xyz[:3], dtype=np.float32).copy()

    @staticmethod
    def _normalize_positive(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(normalized) or normalized <= 0.0:
            return None
        return normalized

    @staticmethod
    def _resolve_depth_m(observation: ObsObject) -> float:
        value = observation.metadata.get("depth_m")
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
        return 0.0


__all__ = [
    "LiveTargetConfig",
    "LiveTargetRequest",
    "LiveTargetService",
    "LiveTargetSnapshot",
]
