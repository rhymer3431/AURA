from __future__ import annotations

from typing import Any

from .pipeline import PerceptionFrameResult


def build_viewer_overlay_payload(
    frame_result: PerceptionFrameResult,
    *,
    max_detections: int = 64,
) -> dict[str, object]:
    projected_by_track: dict[str, dict[str, object]] = {}
    for projected in frame_result.projected_detections:
        track_id = str(projected.track_id)
        entry: dict[str, object] = {}
        metadata = projected.metadata or {}
        depth_value = metadata.get("depth_m")
        if isinstance(depth_value, (int, float)):
            entry["depth_m"] = round(float(depth_value), 4)
        projected_by_track[track_id] = entry

    overlay_detections: list[dict[str, object]] = []
    for tracked in frame_result.tracked_detections[: max(int(max_detections), 0)]:
        detection = tracked.detection
        overlay_entry: dict[str, object] = {
            "class_name": str(detection.class_name),
            "confidence": round(float(detection.confidence), 6),
            "bbox_xyxy": [int(value) for value in detection.bbox_xyxy],
            "track_id": str(tracked.track_id),
        }
        overlay_entry.update(projected_by_track.get(str(tracked.track_id), {}))
        overlay_detections.append(overlay_entry)

    metadata: dict[str, Any] = dict(frame_result.metadata)
    return {
        "detector_backend": str(metadata.get("detector", "")),
        "detector_selected_reason": str(metadata.get("detector_selected_reason", "")),
        "detections": overlay_detections,
    }


__all__ = ["build_viewer_overlay_payload"]
