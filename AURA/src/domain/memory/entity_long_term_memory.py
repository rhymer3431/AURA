import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Set

from src.domain.memory.entity_record import EntityRecord


class EntityLongTermMemory:
    """
    Person-only long-term memory built on ROI (or face) features.

    - 매칭 기준: ROI(proto_feat) 고정, face는 보조 점수로만 사용
    - confirmed 엔티티는 프레임 밖으로 오래 나가도 유지 (장기 기억)
    - prune는 오탐/약한 엔티티 정리 + max_entities 초과 시에만 evict
    - ✅ seen_frames 제거: last_seen_frame + seen_count만 사용
    """

    def __init__(
        self,
        feat_dim: int,
        sim_threshold: float = 0.8,
        iou_threshold: float = 0.2,
        max_entities: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",

        # --- handover / history ---
        handover_window: int = 5,
        max_roi_history: int = 20,

        # --- confirmed 정책 ---
        confirmed_min_seen: int = 12,
        confirmed_min_track: int = 5,

        # --- weak/ephemeral prune ---
        ephemeral_ttl: int = 45,
        suspect_limit: int = 6,
        suspect_ttl: int = 90,

        # --- track mapping 신뢰 만료 ---
        track_mapping_idle_ttl: int = 60,

        # --- re-id 정책 (짧은 공백) ---
        reid_time_window: int = 150,
        reid_iou_threshold: float = 0.05,

        # --- re-id 정책 (장기 공백, appearance-only) ---
        long_reid_sim_threshold: float = 0.90,
        long_reid_face_gate: float = 0.85,

        # --- face 보조 가중치 ---
        face_sim_weight: float = 0.05,
        roi_floor_delta: float = 0.08,

        # --- optional: idle이 길면 히스토리 압축 ---
        archive_after_idle: int = 600,
        archive_keep_history: int = 4,
    ):
        self.feat_dim = feat_dim
        self.sim_threshold = sim_threshold
        self.iou_threshold = iou_threshold
        self.max_entities = max_entities
        self.device = device

        self.handover_window = handover_window
        self.max_roi_history = max_roi_history

        self.confirmed_min_seen = confirmed_min_seen
        self.confirmed_min_track = confirmed_min_track

        self.ephemeral_ttl = ephemeral_ttl
        self.suspect_limit = suspect_limit
        self.suspect_ttl = suspect_ttl

        self.track_mapping_idle_ttl = track_mapping_idle_ttl

        self.reid_iou_threshold = reid_iou_threshold
        self.reid_time_window = reid_time_window

        self.long_reid_sim_threshold = long_reid_sim_threshold
        self.long_reid_face_gate = long_reid_face_gate

        self.face_sim_weight = face_sim_weight
        self.roi_floor_delta = roi_floor_delta

        self.archive_after_idle = archive_after_idle
        self.archive_keep_history = archive_keep_history

        self.entities: Dict[int, EntityRecord] = {}
        self._next_id: int = 1
        self.track_to_entity: Dict[int, int] = {}

    # ------------------- Basic utils -------------------

    def _cosine_sim(self, feat: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        feat = F.normalize(feat, dim=-1)
        protos = F.normalize(protos, dim=-1)
        return (feat @ protos.t()).squeeze(0)

    def _single_cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        return float(F.cosine_similarity(a, b, dim=-1).item())

    def _select_candidates(self, base_cls: str, exclude: Optional[Set[int]] = None) -> List[EntityRecord]:
        if exclude is None:
            exclude = set()
        return [
            e for e in self.entities.values()
            if e.base_cls == base_cls and e.entity_id not in exclude
        ]

    @staticmethod
    def _iou(box1, box2) -> float:
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2
        xa1, ya1 = max(x11, x21), max(y11, y21)
        xa2, ya2 = min(x12, x22), min(y12, y22)
        inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area1 = max(0, x12 - x11) * max(0, y12 - y11)
        area2 = max(0, x22 - x21) * max(0, y22 - y21)
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def _drop_track_mapping_for_entity(self, eid: int):
        to_del = [tid for tid, mapped_eid in self.track_to_entity.items() if mapped_eid == eid]
        for tid in to_del:
            del self.track_to_entity[tid]

    def _is_confirmed(self, ent: EntityRecord) -> bool:
        # ✅ seen_frames 대신 seen_count 사용
        return (ent.seen_count >= self.confirmed_min_seen) or (len(ent.track_history) >= self.confirmed_min_track)

    def _evict_by_priority(self):
        if not self.entities:
            return
        ents = list(self.entities.values())
        ents.sort(key=lambda e: (self._is_confirmed(e), e.last_seen_frame))  # confirmed=False 먼저 제거
        victim = ents[0]
        vid = victim.entity_id
        self._drop_track_mapping_for_entity(vid)
        self.entities.pop(vid, None)

    def _maybe_archive_entity(self, ent: EntityRecord, frame_idx: int):
        dt_idle = frame_idx - ent.last_seen_frame
        if dt_idle < self.archive_after_idle:
            return

        if ent.roi_feat_history and len(ent.roi_feat_history) > self.archive_keep_history:
            ent.roi_feat_history = ent.roi_feat_history[-self.archive_keep_history:]

        if ent.track_history and len(ent.track_history) > self.archive_keep_history:
            ent.track_history = ent.track_history[-self.archive_keep_history:]

        if ent.cls_history and len(ent.cls_history) > 50:
            ent.cls_history = ent.cls_history[-50:]

    # ------------------- Pruning -------------------

    def prune_entities(self, frame_idx: int):
        to_delete: List[int] = []

        for eid, ent in list(self.entities.items()):
            if ent.base_cls != "person":
                to_delete.append(eid)
                continue

            dt_idle = frame_idx - ent.last_seen_frame
            confirmed = self._is_confirmed(ent)

            # 오래 idle이면 track 매핑만 제거
            if dt_idle > self.track_mapping_idle_ttl:
                self._drop_track_mapping_for_entity(eid)

            # confirmed는 삭제하지 않음
            if confirmed:
                self._maybe_archive_entity(ent, frame_idx)
                continue

            # ✅ seen_frames 대신 seen_count 사용
            seen_count = ent.seen_count

            # unconfirmed 약한 엔티티 정리
            if seen_count < max(3, self.confirmed_min_seen // 4):
                if dt_idle > self.ephemeral_ttl:
                    to_delete.append(eid)
                continue

            if ent.suspect_count >= self.suspect_limit and dt_idle > self.suspect_ttl:
                to_delete.append(eid)
                continue

            self._maybe_archive_entity(ent, frame_idx)

        for eid in to_delete:
            self._drop_track_mapping_for_entity(eid)
            self.entities.pop(eid, None)

        while len(self.entities) > self.max_entities:
            self._evict_by_priority()

    # ------------------- Core: Match or Create -------------------

    def match_or_create(
        self,
        roi_feat: torch.Tensor,
        face_feat: Optional[torch.Tensor],
        base_cls: str,
        frame_idx: int,
        box: List[float],
        track_id: Optional[int] = None,
        exclude_ids: Optional[Set[int]] = None,
        meta: Optional[Dict] = None,
    ) -> int:
        if base_cls != "person":
            return -1

        if meta is None:
            meta = {}
        if exclude_ids is None:
            exclude_ids = set()

        roi_feat = roi_feat.to(self.device).detach()
        face_feat = face_feat.to(self.device).detach() if face_feat is not None else None

        best_entity: Optional[EntityRecord] = None
        best_score: float = -1.0
        roi_floor = max(0.0, self.sim_threshold - self.roi_floor_delta)

        # ------------------- 1) Track-based shortcut (ROI 기준) -------------------
        if track_id is not None and track_id >= 0 and track_id in self.track_to_entity:
            mapped_eid = self.track_to_entity[track_id]
            rec = self.entities.get(mapped_eid)
            if rec is not None and rec.base_cls == base_cls and mapped_eid not in exclude_ids:
                dt = frame_idx - rec.last_seen_frame
                if dt <= self.track_mapping_idle_ttl:
                    proto = rec.proto_feat.to(self.device)
                    sim_roi = self._single_cosine(roi_feat, proto)
                    iou_val = self._iou(box, rec.last_box)

                    score = sim_roi
                    if face_feat is not None and rec.last_face_feat is not None:
                        try:
                            sim_face = self._single_cosine(face_feat, rec.last_face_feat.to(self.device))
                            score = sim_roi + self.face_sim_weight * max(0.0, sim_face)
                        except Exception:
                            pass

                    if sim_roi >= self.sim_threshold and iou_val >= self.iou_threshold:
                        best_entity = rec
                        best_score = score
                    else:
                        self.track_to_entity.pop(track_id, None)
                else:
                    self.track_to_entity.pop(track_id, None)

        # ------------------- 2) Appearance search (ROI 기준 + face 보조) -------------------
        candidates = self._select_candidates(base_cls, exclude=exclude_ids)

        if candidates:
            cand_protos: List[torch.Tensor] = []
            ious: List[float] = []
            dts: List[int] = []

            for c in candidates:
                proto = c.proto_feat if c.proto_feat is not None else c.last_roi_feat
                cand_protos.append(proto.to(self.device))
                ious.append(self._iou(box, c.last_box))
                dts.append(frame_idx - c.last_seen_frame)

            cand_protos_t = torch.stack(cand_protos, dim=0).to(self.device)
            sims_roi = self._cosine_sim(roi_feat, cand_protos_t)

            best_idx = None
            best_local = best_score

            for idx, c in enumerate(candidates):
                sim_roi = float(sims_roi[idx].item())
                iou_val = float(ious[idx])
                dt = dts[idx]

                score = sim_roi
                sim_face = None
                if face_feat is not None and c.last_face_feat is not None:
                    try:
                        sim_face = self._single_cosine(face_feat, c.last_face_feat.to(self.device))
                        score = sim_roi + self.face_sim_weight * max(0.0, float(sim_face))
                    except Exception:
                        sim_face = None

                # Handover protection
                if (
                    track_id is not None
                    and track_id not in c.track_history
                    and dt <= self.handover_window
                    and iou_val >= self.iou_threshold * 0.5
                ):
                    continue

                allow_match = False

                # (A) 근접: ROI sim + IoU
                if sim_roi >= self.sim_threshold and iou_val >= self.iou_threshold:
                    allow_match = True

                # (B) 짧은 공백: ROI sim + lenient IoU
                elif (
                    sim_roi >= self.sim_threshold
                    and dt <= self.reid_time_window
                    and iou_val >= self.reid_iou_threshold
                ):
                    allow_match = True

                # (B-보정) ROI가 살짝 부족하지만 face가 매우 강하면 허용
                elif (
                    dt <= self.reid_time_window
                    and sim_roi >= roi_floor
                    and sim_face is not None
                    and float(sim_face) >= self.long_reid_face_gate
                    and iou_val >= self.reid_iou_threshold
                ):
                    allow_match = True

                # (C) 장기 공백: appearance-only(ROI 높게), 또는 face 매우 강하면 보정
                elif dt > self.reid_time_window:
                    if sim_roi >= self.long_reid_sim_threshold:
                        allow_match = True
                    elif (
                        sim_roi >= (self.long_reid_sim_threshold - 0.05)
                        and sim_face is not None
                        and float(sim_face) >= self.long_reid_face_gate
                    ):
                        allow_match = True

                if allow_match and score > best_local:
                    best_local = score
                    best_idx = idx

            if best_idx is not None:
                best_entity = candidates[best_idx]
                best_score = best_local

        # ------------------- 3) Update existing entity -------------------
        if best_entity is not None:
            eid = best_entity.entity_id
            rec = self.entities[eid]

            prev_roi = rec.last_roi_feat.to(self.device) if rec.last_roi_feat is not None else None

            # EMA 업데이트는 ROI로
            alpha = 0.2
            old_proto = rec.proto_feat.to(self.device)
            new_proto = F.normalize((1 - alpha) * old_proto + alpha * roi_feat, dim=-1)
            rec.proto_feat = new_proto.detach().cpu()

            rec.last_roi_feat = roi_feat.detach().cpu()
            if face_feat is not None:
                rec.last_face_feat = face_feat.detach().cpu()

            rec.last_box = box
            rec.last_seen_frame = frame_idx

            # ✅ seen_frames 대신 seen_count++
            rec.seen_count += 1

            rec.roi_feat_history.append(roi_feat.detach().cpu())
            if len(rec.roi_feat_history) > self.max_roi_history:
                rec.roi_feat_history = rec.roi_feat_history[-self.max_roi_history:]

            if track_id is not None and track_id >= 0:
                rec.track_history.append(track_id)
                self.track_to_entity[track_id] = eid

            rec.cls_history.append(base_cls)
            if len(rec.cls_history) > 50:
                rec.cls_history = rec.cls_history[-50:]

            # ROI consistency 체크
            sim_consistency = 1.0
            if prev_roi is not None:
                try:
                    sim_consistency = self._single_cosine(prev_roi, roi_feat)
                except Exception:
                    sim_consistency = 1.0

            if sim_consistency < 0.4:
                rec.suspect_count += 1
            else:
                rec.suspect_count = max(0, rec.suspect_count - 1)

            rec.meta.update(meta)
            return eid

        # ------------------- 4) Create new entity -------------------
        while len(self.entities) >= self.max_entities:
            self._evict_by_priority()

        eid = self._next_id
        self._next_id += 1

        self.entities[eid] = EntityRecord(
            entity_id=eid,
            base_cls=base_cls,

            proto_feat=F.normalize(roi_feat, dim=-1).detach().cpu(),
            last_roi_feat=roi_feat.detach().cpu(),
            last_face_feat=face_feat.detach().cpu() if face_feat is not None else None,

            last_box=box,
            last_seen_frame=frame_idx,
            seen_count=1,  # ✅ 최초 관측 1회

            track_history=[track_id] if (track_id is not None and track_id >= 0) else [],
            cls_history=[base_cls],
            suspect_count=0,
            roi_feat_history=[roi_feat.detach().cpu()],
            meta=meta,
        )

        if track_id is not None and track_id >= 0:
            self.track_to_entity[track_id] = eid

        return eid
