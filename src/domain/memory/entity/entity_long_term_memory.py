import torch
from typing import List, Dict, Optional
from domain.entity_record import EntityRecord
import torch.nn.functional as F
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
    ):
        self.feat_dim = feat_dim
        self.sim_threshold = sim_threshold
        self.iou_threshold = iou_threshold
        self.max_entities = max_entities
        self.device = device

        self.entities: Dict[int, EntityRecord] = {}
        self._next_id: int = 1
        # explicit tracker mapping (e.g., ByteTrack IDs) to keep per-person IDs stable
        self.track_to_entity: Dict[int, int] = {}

    def _cosine_sim(self, feat: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        feat = F.normalize(feat.unsqueeze(0), dim=-1)
        protos = F.normalize(protos, dim=-1)
        return (feat @ protos.t()).squeeze(0)

    def _select_candidates(self, base_cls: str, exclude: Optional[set] = None) -> List[EntityRecord]:
        # same-class filter to avoid collisions between very different categories
        if exclude is None:
            exclude = set()
        return [e for e in self.entities.values() if e.base_cls == base_cls and e.entity_id not in exclude]

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
        self._drop_track_mapping_for_entity(eid)
        del self.entities[eid]

    def prune_entities(self, frame_idx: int):
        """
        자동 엔티티 정리:
        - person이 아닌 base_cls는 즉시 삭제
        - class stability 확인
        - track length 확인
        - embedding consistency 확인
        - last_seen timeout 제거
        """
        to_delete = []

        for eid, ent in list(self.entities.items()):

            # 1) 클래스 mismatch 즉시 제거
            # LTM은 person만 저장해야 한다는 정책 전제
            if ent.base_cls != "person":
                to_delete.append(eid)
                continue

            # 2) class stability: 최근 10 프레임 중 70% 이상 person이어야 함
            if len(ent.cls_history) >= 10:
                recent = ent.cls_history[-10:]
                if recent.count("person") / len(recent) < 0.7:
                    to_delete.append(eid)
                    continue

            # 3) track 길이가 너무 짧으면 삭제 (noise track 제거용)
            track_len = len([t for t in ent.track_history if t is not None])
            if track_len < 5:  # 필요시 튜닝
                to_delete.append(eid)
                continue

            # 4) embedding consistency 검사
            if ent.suspect_count > 3:
                to_delete.append(eid)
                continue

            # 5) last_seen timeout 삭제
            # 사람인데 300프레임 가까이 업데이트 안 되면 버림
            if frame_idx - ent.last_seen_frame > 300:
                to_delete.append(eid)
                continue

        for eid in to_delete:
            self._drop_track_mapping_for_entity(eid)
            self.entities.pop(eid, None)

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
        # LTM에서는 person만 관리하도록 정책화
        if base_cls != "person":
            return -1

        if meta is None:
            meta = {}
        if exclude_ids is None:
            exclude_ids = set()

        roi_feat = roi_feat.to(self.device).detach()
        face_feat = face_feat.to(self.device).detach() if face_feat is not None else None
        candidates = self._select_candidates(base_cls, exclude=exclude_ids)

        best_entity = None
        best_sim = -1.0

        # 1) If a tracker ID is provided and already known, reuse its entity
        if track_id is not None and track_id >= 0 and track_id in self.track_to_entity:
            mapped_eid = self.track_to_entity[track_id]
            if mapped_eid in self.entities:
                rec = self.entities[mapped_eid]
                if rec.base_cls == base_cls:
                    if mapped_eid in exclude_ids:
                        rec = None
                    else:
                        best_entity = rec
                        best_sim = float(self._cosine_sim(roi_feat, rec.last_roi_feat.to(self.device)).item())

        # 2) Otherwise, fallback to similarity + IoU gating within same class
        query_feat = face_feat if (face_feat is not None and base_cls == "person") else roi_feat

        if candidates and best_entity is None:
            cand_feats = []
            for c in candidates:
                if base_cls == "person" and c.last_face_feat is not None:
                    cand_feats.append(c.last_face_feat)
                else:
                    cand_feats.append(c.last_roi_feat)
            cand_feats_t = torch.stack(cand_feats, dim=0).to(self.device)
            sims = self._cosine_sim(query_feat, cand_feats_t)
            ious = [self._iou(box, c.last_box) for c in candidates]
            best_idx = int(torch.argmax(sims).item())
            best_sim = float(sims[best_idx].item())
            best_iou = float(ious[best_idx])
            candidate = candidates[best_idx]
            # accept if visually similar AND spatially overlapping
            if best_sim >= self.sim_threshold and best_iou >= self.iou_threshold:
                best_entity = candidate

        # 3) 기존 entity 업데이트
        if best_entity is not None and best_sim >= self.sim_threshold:
            eid = best_entity.entity_id
            rec = self.entities[eid]

            # prototype 업데이트
            alpha = 0.2
            new_proto = F.normalize(
                (1 - alpha) * rec.proto_feat.to(self.device) + alpha * roi_feat,
                dim=-1,
            )
            rec.proto_feat = new_proto.detach().cpu()
            rec.last_roi_feat = roi_feat.detach().cpu()
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

            # embedding consistency 업데이트
            try:
                sim_for_consistency = float(
                    F.cosine_similarity(
                        rec.last_roi_feat.to(self.device),
                        roi_feat,
                        dim=-1,
                    ).item()
                )
            except Exception:
                sim_for_consistency = 1.0  # 에러시 일단 안전하게 처리

            if sim_for_consistency < 0.4:
                rec.suspect_count += 1
            else:
                rec.suspect_count = max(0, rec.suspect_count - 1)

            rec.meta.update(meta)
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
        )
        if track_id is not None and track_id >= 0:
            self.track_to_entity[track_id] = eid
        return eid