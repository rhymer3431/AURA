from __future__ import annotations

from dataclasses import dataclass

from inference.detectors.base import DetectionResult


@dataclass(frozen=True)
class TrackedDetection:
    track_id: str
    detection: DetectionResult


class SimpleTrackManager:
    def __init__(self, *, max_distance_px: float = 64.0) -> None:
        self._max_distance_px = float(max_distance_px)
        self._tracks: dict[str, tuple[str, tuple[float, float]]] = {}
        self._track_seq = 0

    def update(self, detections: list[DetectionResult]) -> list[TrackedDetection]:
        assigned: list[TrackedDetection] = []
        next_tracks: dict[str, tuple[str, tuple[float, float]]] = {}
        for detection in detections:
            centroid = detection.centroid_xy or self._bbox_center(detection.bbox_xyxy)
            track_id = detection.track_hint or self._match_track(detection.class_name, centroid)
            next_tracks[track_id] = (detection.class_name, centroid)
            assigned.append(TrackedDetection(track_id=track_id, detection=detection))
        self._tracks = next_tracks
        return assigned

    def _match_track(self, class_name: str, centroid: tuple[float, float]) -> str:
        best_track = ""
        best_distance = float("inf")
        for track_id, (track_class, track_centroid) in self._tracks.items():
            if track_class != class_name:
                continue
            dx = float(track_centroid[0]) - float(centroid[0])
            dy = float(track_centroid[1]) - float(centroid[1])
            distance = (dx * dx + dy * dy) ** 0.5
            if distance < self._max_distance_px and distance < best_distance:
                best_track = track_id
                best_distance = distance
        if best_track != "":
            return best_track
        self._track_seq += 1
        return f"{class_name}_{self._track_seq:04d}"

    @staticmethod
    def _bbox_center(bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
        x0, y0, x1, y1 = bbox_xyxy
        return ((float(x0) + float(x1)) * 0.5, (float(y0) + float(y1)) * 0.5)
