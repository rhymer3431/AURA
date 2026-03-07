from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from inference.detectors.base import DetectionResult, DetectorBackend
from inference.detectors.factory import DetectorFactoryConfig, create_detector_backend
from inference.trackers.simple_tracker import SimpleTrackManager, TrackedDetection
from memory.models import ObsObject

from .depth_projection import DepthProjector, ProjectedDetection
from .object_mapper import Detection2D, ObjectMapper
from .observation_fuser import ObservationFuser
from .speaker_events import SpeakerEvent


@dataclass(frozen=True)
class PerceptionFrameResult:
    detections: list[DetectionResult]
    tracked_detections: list[TrackedDetection]
    projected_detections: list[ProjectedDetection]
    observations: list[ObsObject]
    speaker_events: list[SpeakerEvent]
    metadata: dict[str, Any]


class PerceptionPipeline:
    def __init__(
        self,
        detector: DetectorBackend | None = None,
        *,
        detector_config: DetectorFactoryConfig | None = None,
        tracker: SimpleTrackManager | None = None,
        projector: DepthProjector | None = None,
        mapper: ObjectMapper | None = None,
        fuser: ObservationFuser | None = None,
    ) -> None:
        self.detector = detector or create_detector_backend(detector_config)
        self.tracker = tracker or SimpleTrackManager()
        self.projector = projector or DepthProjector()
        self.mapper = mapper or ObjectMapper()
        self.fuser = fuser or ObservationFuser()

    def process_frame(
        self,
        *,
        rgb_image: np.ndarray,
        depth_image_m: np.ndarray,
        timestamp: float,
        camera_pose_xyz: tuple[float, float, float] | np.ndarray,
        camera_quat_wxyz: tuple[float, float, float, float] | np.ndarray,
        camera_intrinsic: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> PerceptionFrameResult:
        metadata = dict(metadata or {})
        detections = self.detector.detect(rgb_image, timestamp=timestamp, metadata=metadata)
        tracked = self.tracker.update(detections)
        projected: list[ProjectedDetection] = []
        observations: list[ObsObject] = []
        for tracked_detection in tracked:
            projected_detection = self.projector.project(
                tracked_detection,
                depth_image_m=depth_image_m,
                camera_intrinsic=camera_intrinsic,
                camera_pose_xyz=camera_pose_xyz,
                camera_quat_wxyz=camera_quat_wxyz,
                metadata=metadata,
            )
            if projected_detection is None:
                continue
            projected.append(projected_detection)
            detection2d = Detection2D(
                class_name=projected_detection.class_name,
                world_pose_xyz=projected_detection.world_pose_xyz,
                timestamp=timestamp,
                confidence=projected_detection.confidence,
                track_id=projected_detection.track_id,
                room_id=projected_detection.room_id,
                movable=projected_detection.movable,
                state=projected_detection.state,
                embedding_id=projected_detection.embedding_id,
                snapshots=list(projected_detection.snapshots or []),
                metadata=dict(projected_detection.metadata or {}),
            )
            observations.append(self.mapper.to_obs_object(detection2d))
        fused = self.fuser.fuse(observations)
        speaker_events = self._parse_speaker_events(metadata, timestamp)
        return PerceptionFrameResult(
            detections=detections,
            tracked_detections=tracked,
            projected_detections=projected,
            observations=fused,
            speaker_events=speaker_events,
            metadata={"detector": self.detector.info.backend_name, **metadata},
        )

    @staticmethod
    def _parse_speaker_events(metadata: dict[str, Any], timestamp: float) -> list[SpeakerEvent]:
        raw_events = metadata.get("speaker_events", [])
        results: list[SpeakerEvent] = []
        if not isinstance(raw_events, list):
            return results
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            results.append(
                SpeakerEvent(
                    timestamp=float(item.get("timestamp", timestamp)),
                    direction_yaw_rad=float(item.get("direction_yaw_rad", 0.0)),
                    speaker_id=str(item.get("speaker_id", "")),
                    confidence=float(item.get("confidence", 1.0)),
                    metadata={key: value for key, value in item.items() if key not in {"timestamp", "direction_yaw_rad", "speaker_id", "confidence"}},
                )
            )
        return results
