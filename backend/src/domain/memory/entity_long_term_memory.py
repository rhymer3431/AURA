import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from domain.memory.entity_record import EntityRecord
from infrastructure.logging.pipeline_logger import PipelineLogger


class EntityLongTermMemory:
    """
    ROI feature + base class 로 entity_id를 할당/유지하는 LTM.
    (정책: person 전용 LTM)
    """

    def __init__(
        self,
        feat_dim: int,
        sim_threshold: float = 0.8,
        iou_threshold: float = 0.2,
        max_entities: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[PipelineLogger] = None,
        # --- 재등장/프루닝 튜닝 파라미터 ---
        reid_sim_ratio: float = 0.7,       # 재등장용 sim 임계값 비율 (sim_threshold * reid_sim_ratio)
        near_window: int = 30,             # IoU를 강하게 보는 근접 프레임 수
        prune_short_tracks: bool = False,  # 짧은 트랙 삭제 여부 (테스트 기본: 비활성)
        min_track_length: int = 5,
        prune_timeout_frames: int = 0,     # 0이면 timeout으로 삭제 안 함
    ):
        self.feat_dim = feat_dim
        self.sim_threshold = sim_threshold
        self.iou_threshold = iou_threshold
        self.max_entities = max_entities
        self.device = device
        self.logger = logger
        self.max_roi_history = 20

        # 재등장 튜닝
        self.reid_sim_threshold = sim_threshold * reid_sim_ratio
        self.near_window = near_window

        # 프루닝 튜닝
        self.prune_short_tracks = prune_short_tracks
        self.min_track_length = min_track_length
        self.prune_timeout_frames = prune_timeout_frames

        self.entities: Dict[int, EntityRecord] = {}
        self._next_id: int = 1
        # explicit tracker mapping (e.g., ByteTrack IDs) to keep per-person IDs stable
        self.track_to_entity: Dict[int, int] = {}

    # ---------------------------------------------------------------------
    # 내부 유틸
    # ---------------------------------------------------------------------
    def _log(self, event: str, frame_idx: Optional[int] = None, level: str = "INFO", **payload):
        if self.logger is not None:
            self.logger.log(
                module="EntityLTM",
                event=event,
                frame_idx=frame_idx,
                level=level,
                **payload,
            )

    def _cosine_sim(self, feat: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        feat = F.normalize(feat.unsqueeze(0), dim=-1)
        protos = F.normalize(protos, dim=-1)
        return (feat @ protos.t()).squeeze(0)

    def _select_candidates(self, base_cls: str, exclude: Optional[set] = None) -> List[EntityRecord]:
        # same-class filter to avoid collisions between very different categories
        if exclude is None:
            exclude = set()
        return [e for e in self.entities.values() if e.base_cls == base_cls and e.entity_id not in exclude]

    def _best_roi_similarity(self, rec: EntityRecord, roi_query: torch.Tensor) -> Optional[float]:
        if not rec.roi_feat_history:
            feats = rec.last_roi_feat.unsqueeze(0).to(self.device)
        else:
            feats = torch.stack(rec.roi_feat_history, dim=0).to(self.device)
        try:
            sims = self._cosine_sim(roi_query, feats)
            return float(torch.max(sims).item())
        except Exception:
            return None

    def _face_similarity(self, rec: EntityRecord, face_query: Optional[torch.Tensor]) -> Optional[float]:
        if face_query is None or rec.last_face_feat is None:
            return None
        try:
            return float(
                F.cosine_similarity(
                    face_query.unsqueeze(0),
                    rec.last_face_feat.unsqueeze(0).to(self.device),
                    dim=-1,
                ).item()
            )
        except Exception:
            return None

    def _append_roi_history(self, rec: EntityRecord, roi_feat: torch.Tensor):
        rec.roi_feat_history.append(roi_feat.detach().cpu())
        if len(rec.roi_feat_history) > self.max_roi_history:
            rec.roi_feat_history = rec.roi_feat_history[-self.max_roi_history:]

    @staticmethod
    def _iou(box1, box2):
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
        """해당 entity_id를 참조하는 track_to_entity 매핑 정리."""
        to_del = []
        for tid, mapped_eid in self.track_to_entity.items():
            if mapped_eid == eid:
                to_del.append(tid)
        for tid in to_del:
            del self.track_to_entity[tid]

    def _evict_oldest(self):
        if not self.entities:
            return
        oldest = min(self.entities.values(), key=lambda e: e.last_seen_frame)
        eid = oldest.entity_id
        track_id = oldest.track_history[-1] if oldest.track_history else None
        self._drop_track_mapping_for_entity(eid)
        del self.entities[eid]
        self._log(
            event="delete",
            frame_idx=oldest.last_seen_frame,
            reason="evict_oldest",
            entity_id=eid,
            track_id=track_id,
            base_cls=oldest.base_cls,
        )

    def _accept_match(self, sim: float, iou: Optional[float], dt: Optional[int]) -> bool:
        """
        매칭 허용 여부를 결정하는 게이트:
        - 가까운 과거(near_window 이내): sim + IoU 모두 사용
        - 먼 과거(재등장): IoU 없이 완화된 sim 임계값 사용
        """
        if dt is None:
            dt = 0
        # 근접 프레임: 위치 관계까지 강하게 본다
        if dt <= self.near_window:
            return (
                sim >= self.sim_threshold
                and iou is not None
                and iou >= self.iou_threshold
            )
        # 재등장 구간: appearance 위주로 판단
        return sim >= self.reid_sim_threshold

    # ---------------------------------------------------------------------
    # 엔티티 정리(prune)
    # ---------------------------------------------------------------------
    def prune_entities(self, frame_idx: int):
        """
        Periodic pruning for stale or noisy entities.
        튜닝 포인트:
        - 기본 설정에서는 short_track / timeout 삭제를 거의 하지 않도록 완화.
        """
        to_delete: List[int] = []

        for eid, ent in list(self.entities.items()):
            last_track = ent.track_history[-1] if ent.track_history else None

            # 1) person이 아닌 엔티티는 즉시 삭제
            if ent.base_cls != "person":
                to_delete.append(eid)
                self._log(
                    event="delete",
                    frame_idx=frame_idx,
                    entity_id=eid,
                    track_id=last_track,
                    base_cls=ent.base_cls,
                    reason="non_person",
                )
                continue

            # 2) class stability 체크
            if len(ent.cls_history) >= 10:
                recent = ent.cls_history[-10:]
                if recent.count("person") / len(recent) < 0.7:
                    to_delete.append(eid)
                    self._log(
                        event="delete",
                        frame_idx=frame_idx,
                        entity_id=eid,
                        track_id=last_track,
                        base_cls=ent.base_cls,
                        reason="unstable_class",
                    )
                    continue

            # 3) 짧은 트랙 삭제 (옵션)
            if self.prune_short_tracks:
                track_len = len([t for t in ent.track_history if t is not None])
                if track_len < self.min_track_length:
                    to_delete.append(eid)
                    self._log(
                        event="delete",
                        frame_idx=frame_idx,
                        entity_id=eid,
                        track_id=last_track,
                        base_cls=ent.base_cls,
                        reason="short_track",
                    )
                    continue

            # 4) 임베딩 일관성 문제
            if ent.suspect_count > 3:
                to_delete.append(eid)
                self._log(
                    event="delete",
                    frame_idx=frame_idx,
                    entity_id=eid,
                    track_id=last_track,
                    base_cls=ent.base_cls,
                    reason="inconsistent_embedding",
                )
                continue

            # 5) timeout (옵션)
            if self.prune_timeout_frames > 0:
                if frame_idx - ent.last_seen_frame > self.prune_timeout_frames:
                    to_delete.append(eid)
                    self._log(
                        event="delete",
                        frame_idx=frame_idx,
                        entity_id=eid,
                        track_id=last_track,
                        base_cls=ent.base_cls,
                        reason="timeout",
                    )
                    continue

        for eid in to_delete:
            self._drop_track_mapping_for_entity(eid)
            self.entities.pop(eid, None)

    # ---------------------------------------------------------------------
    # 매칭 / 엔티티 생성
    # ---------------------------------------------------------------------
    def match_or_create(
        self,
        roi_feat: torch.Tensor,
        face_feat: Optional[torch.Tensor],
        base_cls: str,
        frame_idx: int,
        box: List[float],
        track_id: Optional[int] = None,
        exclude_ids: Optional[set] = None,
        meta: Optional[Dict] = None,
    ) -> int:
        # LTM은 person만 관리
        if base_cls != "person":
            return -1

        if meta is None:
            meta = {}
        if exclude_ids is None:
            exclude_ids = set()

        roi_feat = roi_feat.to(self.device).detach()
        face_feat = face_feat.to(self.device).detach() if face_feat is not None else None
        candidates = self._select_candidates(base_cls, exclude=exclude_ids)

        best_entity: Optional[EntityRecord] = None
        best_sim: float = -1.0
        best_iou: Optional[float] = None
        best_dt: Optional[int] = None

        def combined_similarity(rec: EntityRecord) -> Optional[float]:
            sims: List[float] = []
            roi_sim = self._best_roi_similarity(rec, roi_feat)
            if roi_sim is not None:
                sims.append(roi_sim)
            face_sim = self._face_similarity(rec, face_feat)
            if face_sim is not None:
                sims.append(face_sim)
            if not sims:
                return None
            return max(sims)

        # 1) tracker 기반 재사용 시도
        if track_id is not None and track_id >= 0 and track_id in self.track_to_entity:
            mapped_eid = self.track_to_entity[track_id]
            if mapped_eid in self.entities:
                rec = self.entities[mapped_eid]
                if rec.base_cls == base_cls and mapped_eid not in exclude_ids:
                    sim_val = combined_similarity(rec)
                    if sim_val is not None:
                        iou_val = self._iou(box, rec.last_box)
                        dt = frame_idx - rec.last_seen_frame
                        if self._accept_match(sim_val, iou_val, dt):
                            best_entity = rec
                            best_sim = sim_val
                            best_iou = iou_val
                            best_dt = dt

        # 2) tracker 매칭 실패 시, 모든 후보 대상으로 fallback
        if candidates and best_entity is None:
            scored: List[Tuple[EntityRecord, float, float, int]] = []
            for c in candidates:
                sim_val = combined_similarity(c)
                if sim_val is None:
                    continue
                iou_val = self._iou(box, c.last_box)
                dt = frame_idx - c.last_seen_frame
                scored.append((c, sim_val, iou_val, dt))

            if scored:
                cand, cand_sim, cand_iou, cand_dt = max(scored, key=lambda x: x[1])
                if self._accept_match(cand_sim, cand_iou, cand_dt):
                    best_entity = cand
                    best_sim = cand_sim
                    best_iou = cand_iou
                    best_dt = cand_dt

        # 3) 기존 entity 업데이트
        if best_entity is not None:
            eid = best_entity.entity_id
            rec = self.entities[eid]

            # 이전 ROI를 보관해 두고 일관성 체크에 사용
            prev_roi = rec.last_roi_feat.to(self.device)

            # prototype 업데이트
            alpha = 0.2
            new_proto = F.normalize(
                (1 - alpha) * rec.proto_feat.to(self.device) + alpha * roi_feat,
                dim=-1,
            )
            rec.proto_feat = new_proto.detach().cpu()
            rec.last_roi_feat = roi_feat.detach().cpu()
            self._append_roi_history(rec, roi_feat)
            if face_feat is not None and base_cls == "person":
                rec.last_face_feat = face_feat.detach().cpu()
            rec.last_box = box
            rec.last_seen_frame = frame_idx
            rec.seen_frames.append(frame_idx)

            # track 기록
            if track_id is not None and track_id >= 0:
                rec.track_history.append(track_id)
                self.track_to_entity[track_id] = eid

            # class history 업데이트
            rec.cls_history.append(base_cls)
            if len(rec.cls_history) > 50:
                rec.cls_history = rec.cls_history[-50:]

            # embedding consistency 업데이트 (이전 ROI vs 새 ROI)
            try:
                sim_for_consistency = float(
                    F.cosine_similarity(
                        prev_roi,
                        roi_feat,
                        dim=-1,
                    ).item()
                )
            except Exception:
                sim_for_consistency = 1.0  # 에러시 안전하게 처리

            if sim_for_consistency < 0.4:
                rec.suspect_count += 1
            else:
                rec.suspect_count = max(0, rec.suspect_count - 1)

            rec.meta.update(meta)
            self._log(
                event="update",
                frame_idx=frame_idx,
                entity_id=eid,
                track_id=track_id,
                base_cls=base_cls,
                sim=best_sim,
                iou=best_iou,
                dt=best_dt,
            )
            return eid

        # 4) 새 entity 생성
        if len(self.entities) >= self.max_entities:
            self._evict_oldest()

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
            seen_frames=[frame_idx],
            track_history=[track_id] if track_id is not None else [],
            meta=meta,
            cls_history=[base_cls],
            suspect_count=0,
            roi_feat_history=[roi_feat.detach().cpu()],
        )
        if track_id is not None and track_id >= 0:
            self.track_to_entity[track_id] = eid
        self._log(
            event="create",
            frame_idx=frame_idx,
            entity_id=eid,
            track_id=track_id,
            base_cls=base_cls,
            sim=best_sim if best_sim is not None and best_sim >= 0 else None,
            iou=best_iou,
            dt=None,
        )
        return eid
