from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from .contracts import Detection2D3D, ObjectMemoryEntry, Pose2D, pose_to_dict


class SceneMemory:
    """STM/LTM memory with map-anchored object entries."""

    def __init__(self, cfg: Dict) -> None:
        self.stm_window_s = float(cfg.get("stm_window_s", 8.0))
        self.stm_max_entries = int(cfg.get("stm_max_entries", 128))
        self.ltm_threshold = float(cfg.get("ltm_threshold", 0.55))

        self._weights = {
            "task_relevance": float(cfg.get("w_task_relevance", 0.35)),
            "frequency": float(cfg.get("w_frequency", 0.20)),
            "recency": float(cfg.get("w_recency", 0.20)),
            "mentioned": float(cfg.get("w_mentioned", 0.15)),
            "confidence": float(cfg.get("w_confidence", 0.10)),
        }

        self._lock = threading.Lock()
        self._stm: Deque[ObjectMemoryEntry] = deque(maxlen=self.stm_max_entries)
        self._ltm_by_id: Dict[str, ObjectMemoryEntry] = {}
        self._sightings: Dict[str, int] = {}
        self._task_focus: set[str] = set()
        self._mentioned: set[str] = set()
        self._start_pose: Optional[Pose2D] = None

    def set_task_focus(self, object_names: List[str]) -> None:
        with self._lock:
            self._task_focus = {name.strip().lower() for name in object_names if name}
            self._mentioned = set(self._task_focus)

    def set_start_pose(self, pose: Pose2D) -> None:
        with self._lock:
            self._start_pose = pose

    def get_start_pose(self) -> Optional[Pose2D]:
        with self._lock:
            return self._start_pose

    def _map_anchor_pose(self, detection: Detection2D3D, robot_pose: Optional[Pose2D]) -> Pose2D:
        if detection.position_in_map is not None:
            return detection.position_in_map

        base = robot_pose or Pose2D(x=0.0, y=0.0, yaw=0.0)
        bbox = detection.bbox_xywh
        depth_hint = 1.0
        if bbox is not None:
            depth_hint = max(0.4, min(2.0, bbox[2] / 120.0))
        return Pose2D(
            x=base.x + depth_hint * math.cos(base.yaw),
            y=base.y + depth_hint * math.sin(base.yaw),
            yaw=base.yaw,
            frame_id="map",
            covariance_norm=base.covariance_norm,
        )

    def _importance(self, class_name: str, object_id: str, confidence: float, timestamp: float) -> float:
        now = time.time()
        age = max(0.0, now - timestamp)
        recency = math.exp(-age / max(1e-3, self.stm_window_s))
        frequency = min(1.0, self._sightings.get(object_id, 0) / 6.0)
        task_relevance = 1.0 if class_name in self._task_focus else 0.0
        mentioned = 1.0 if class_name in self._mentioned else 0.0

        score = (
            self._weights["task_relevance"] * task_relevance
            + self._weights["frequency"] * frequency
            + self._weights["recency"] * recency
            + self._weights["mentioned"] * mentioned
            + self._weights["confidence"] * max(0.0, min(1.0, confidence))
        )
        return max(0.0, min(1.0, score))

    def update_from_detection(
        self, detections: List[Detection2D3D], robot_pose: Optional[Pose2D]
    ) -> None:
        now = time.time()
        with self._lock:
            for det in detections:
                class_name = det.class_name.strip().lower()
                object_id = det.object_id.strip().lower() or f"{class_name}-{int(now*1000)}"
                self._sightings[object_id] = self._sightings.get(object_id, 0) + 1
                anchored_pose = self._map_anchor_pose(det, robot_pose)
                importance = self._importance(class_name, object_id, det.score, det.timestamp)

                entry = ObjectMemoryEntry(
                    object_id=object_id,
                    class_name=class_name,
                    map_pose=anchored_pose,
                    last_seen=det.timestamp,
                    confidence=max(0.0, min(1.0, det.score)),
                    importance=importance,
                )
                self._stm.append(entry)
                if importance >= self.ltm_threshold:
                    self._ltm_by_id[object_id] = entry

            # Sliding window trim by time, not only by count.
            while self._stm and (now - self._stm[0].last_seen) > self.stm_window_s:
                self._stm.popleft()

    def get_object_pose(self, key: str) -> Optional[Tuple[Pose2D, float, float]]:
        key = key.strip().lower()
        with self._lock:
            if key in self._ltm_by_id:
                entry = self._ltm_by_id[key]
                return entry.map_pose, entry.confidence, entry.last_seen

            # Fall back to class lookup from LTM newest entry.
            candidates = [v for v in self._ltm_by_id.values() if v.class_name == key]
            if candidates:
                newest = max(candidates, key=lambda e: e.last_seen)
                return newest.map_pose, newest.confidence, newest.last_seen

            # Fall back to STM when no long-term anchor is available yet.
            stm_candidates = [v for v in self._stm if v.class_name == key or v.object_id == key]
            if stm_candidates:
                newest = max(stm_candidates, key=lambda e: e.last_seen)
                return newest.map_pose, newest.confidence, newest.last_seen
        return None

    def summary(self, max_objects: int = 8) -> Dict:
        with self._lock:
            ltm_sorted = sorted(self._ltm_by_id.values(), key=lambda e: e.last_seen, reverse=True)
            objects = [
                {
                    "object_id": e.object_id,
                    "class": e.class_name,
                    "map_pose": pose_to_dict(e.map_pose),
                    "last_seen": e.last_seen,
                    "confidence": e.confidence,
                    "importance": e.importance,
                }
                for e in ltm_sorted[:max_objects]
            ]
            return {
                "stm_size": len(self._stm),
                "ltm_size": len(self._ltm_by_id),
                "task_focus": sorted(self._task_focus),
                "objects": objects,
                "start_pose": pose_to_dict(self._start_pose),
            }

