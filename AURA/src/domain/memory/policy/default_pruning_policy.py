# src/domain/memory/default_pruning_policy.py
from typing import (Tuple)
from .pruning_policy import PruningPolicy
from src.domain.memory.entity_record import EntityRecord

class DefaultPruningPolicy(PruningPolicy):

    def __init__(
        self,
        min_track_length: int = 5,
        prune_timeout_frames: int = 0,
        unstable_cls_window: int = 10,
        unstable_cls_ratio: float = 0.7,
        suspect_threshold: int = 3,
    ):
        self.min_track_length = min_track_length
        self.prune_timeout_frames = prune_timeout_frames
        self.unstable_cls_window = unstable_cls_window
        self.unstable_cls_ratio = unstable_cls_ratio
        self.suspect_threshold = suspect_threshold

    def should_delete(
        self,
        ent: EntityRecord,
        frame_idx: int,
    ) -> Tuple[bool, str]:

        last_track = ent.track_history[-1] if ent.track_history else None

        # 1) 클래스 필터 (person만 유지)
        if ent.base_cls != "person":
            return True, "non_person"

        # 2) 최근 class 안정성 체크
        if len(ent.cls_history) >= self.unstable_cls_window:
            recent = ent.cls_history[-self.unstable_cls_window:]
            ratio = recent.count("person") / len(recent)
            if ratio < self.unstable_cls_ratio:
                return True, "unstable_class"

        # 3) track length 조건
        track_len = len([t for t in ent.track_history if t is not None])
        if track_len < self.min_track_length:
            return True, "short_track"

        # 4) 임베딩 불일치 카운트
        if ent.suspect_count > self.suspect_threshold:
            return True, "inconsistent_embedding"

        # 5) timeout 조건
        if self.prune_timeout_frames > 0:
            if frame_idx - ent.last_seen_frame > self.prune_timeout_frames:
                return True, "timeout"

        return False, ""
