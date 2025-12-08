from typing import Dict, List, Tuple, Optional, Iterable, Any
from collections import Counter
import math

import torch

from domain.memory.entity_record import EntityRecord
from domain.memory.entity_long_term_memory import EntityLongTermMemory
from domain.memory.short_term_graph_memory import ShortTermGraphMemory
from domain.perception.entity_node import EntityNode
from domain.perception.scene_graph_frame import SceneGraphFrame
from domain.perception.simple_scene_graph_frame import SimpleSceneGraphFrame
from infrastructure.perception.build_grin_input import build_grin_input
from infrastructure.perception.grin_stub_model import GRINStubModel
from infrastructure.perception.yolo_world_detector import YoloWorldDetector
from infrastructure.logging.pipeline_logger import PipelineLogger


class PerceptionServiceAdapter:
    """Adapter that wraps the YOLO-world + LTM perception system to satisfy PerceptionPort."""

    # --- Simple relation id mapping (rule-based) ---
    RELATION_TYPES: Dict[str, int] = {
        "left_of": 0,
        "right_of": 1,
        "above": 2,
        "below": 3,
        "overlaps": 4,
        "near": 5,
    }
    RELATION_ID_TO_NAME: Dict[int, str] = {
        v: k for k, v in RELATION_TYPES.items()
    }

    def __init__(
        self,
        yolo_weight: str = "yolov8s-worldv2.pt",
        ltm_feat_dim: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[PipelineLogger] = None,
    ):
        self.device = device
        self.logger = logger
        self.detector = YoloWorldDetector(
            weight=yolo_weight,
            device=device,
            logger=logger,
        )
        self.ltm = EntityLongTermMemory(
            feat_dim=ltm_feat_dim,
            sim_threshold=0.7,
            max_entities=512,
            device=device,
            logger=logger,
        )
        self.stm = ShortTermGraphMemory(max_frames=60, logger=logger)
        self.grin = GRINStubModel(node_dim=ltm_feat_dim).to(device)

    def _log(
        self, event: str, frame_idx: Optional[int] = None, level: str = "INFO", **payload
    ):
        if self.logger is not None:
            self.logger.log(
                module="PerceptionService",
                event=event,
                frame_idx=frame_idx,
                level=level,
                **payload,
            )

    def build_simple_scene_graph_frame(
        self,
        sg_frame: SceneGraphFrame,
    ) -> SimpleSceneGraphFrame:
        # 노드가 하나도 없으면 완전 빈 프레임 리턴
        if not sg_frame.nodes:
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            empty_scores = torch.zeros((0,), dtype=torch.float32)
            return SimpleSceneGraphFrame(
                frame_idx=sg_frame.frame_idx,
                boxes=empty_boxes,
                labels=[],
                scores=empty_scores,
                track_ids=[],
                entity_ids=[],
                static_pairs=torch.zeros((0, 2), dtype=torch.long),
                static_rel_names=[],
                temporal_pairs=torch.zeros((0, 2), dtype=torch.long),
                temporal_rel_names=[],
            )

        boxes: List[List[float]] = []
        labels: List[str] = []
        scores: List[float] = []
        track_ids: List[int] = []
        entity_ids: List[int] = []

        for n in sg_frame.nodes:
            boxes.append(n.box)
            labels.append(n.cls)
            scores.append(float(getattr(n, "score", 1.0)))
            track_ids.append(n.track_id if n.track_id is not None else -1)

            # 일단 person만 persistent entity로 취급 (나머지는 -1)
            if n.cls == "person" and n.entity_id is not None and n.entity_id >= 0:
                entity_ids.append(int(n.entity_id))
            else:
                entity_ids.append(-1)

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)

        # --- relations → static_pairs / static_rel_names 로 풀어주기 ---
        static_pairs_list: List[Tuple[int, int]] = []
        static_rel_names: List[str] = []

        for sub_idx, obj_idx, rel_id in getattr(sg_frame, "relations", []):
            static_pairs_list.append((sub_idx, obj_idx))
            static_rel_names.append(
                self.RELATION_ID_TO_NAME.get(rel_id, "unknown")
            )

        if static_pairs_list:
            static_pairs_t = torch.tensor(static_pairs_list, dtype=torch.long)
        else:
            static_pairs_t = torch.zeros((0, 2), dtype=torch.long)

        return SimpleSceneGraphFrame(
            frame_idx=sg_frame.frame_idx,
            boxes=boxes_t,
            labels=labels,
            scores=scores_t,
            track_ids=track_ids,
            entity_ids=entity_ids,
            static_pairs=static_pairs_t,
            static_rel_names=static_rel_names,
            # 현재는 temporal relation rule-base 미구현
            temporal_pairs=torch.zeros((0, 2), dtype=torch.long),
            temporal_rel_names=[],
        )

    def _assign_entities_with_ltm(
        self,
        detections: List[Dict],
        frame_idx: int,
    ) -> List[EntityNode]:
        nodes: List[EntityNode] = []
        used_entities: set[int] = set()
        track_ids = [det.get("track_id") for det in detections]
        dup_track_ids = {
            tid for tid, cnt in Counter(track_ids).items() if tid is not None and cnt > 1
        }
        for det in detections:
            base_cls = det["cls"]
            roi_feat = det["roi_feat"]
            face_feat = det.get("face_feat")
            raw_track_id = det["track_id"]
            # If the tracker assigns the same ID to multiple detections in a frame,
            # ignore it to avoid merging distinct entities.
            if raw_track_id in dup_track_ids or raw_track_id is None or raw_track_id < 0:
                track_id = None
            else:
                track_id = raw_track_id
            box = det["box"]
            score = float(det["score"])

            eid = self.ltm.match_or_create(
                roi_feat=roi_feat,
                face_feat=face_feat,
                base_cls=base_cls,
                frame_idx=frame_idx,
                box=box,
                track_id=track_id,
                exclude_ids=used_entities,
            )
            used_entities.add(eid)

            node = EntityNode(
                entity_id=eid,
                track_id=track_id,
                cls=base_cls,
                box=box,
                roi_feat=roi_feat,
                frame_idx=frame_idx,
                score=score,
            )
            nodes.append(node)

        return nodes

    # -----------------------
    # Rule-based relation SGG
    # -----------------------
    @staticmethod
    def _box_center_and_size(box: Iterable[float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return cx, cy, w, h

    @staticmethod
    def _box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))

        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom

    def _build_relations_stub(self, nodes: List[EntityNode]) -> List[Tuple[int, int, int]]:
        """
        간단한 rule-based static relation:
        - left_of / right_of: 중심 x좌표 비교
        - above / below: 중심 y좌표 비교
        - overlaps: IoU 기준
        - near: 중심 거리 / 박스 크기 기준
        """
        relations: List[Tuple[int, int, int]] = []
        n = len(nodes)
        if n < 2:
            return relations

        for i in range(n):
            ci_x, ci_y, ci_w, ci_h = self._box_center_and_size(nodes[i].box)
            for j in range(i + 1, n):
                cj_x, cj_y, cj_w, cj_h = self._box_center_and_size(nodes[j].box)

                # --- left/right ---
                # 약간의 margin을 두고 한쪽으로 충분히 치우친 경우만 관계 생성
                margin_x = 0.05 * (ci_w + cj_w)
                if ci_x + margin_x < cj_x:
                    relations.append((i, j, self.RELATION_TYPES["left_of"]))
                    relations.append((j, i, self.RELATION_TYPES["right_of"]))
                elif cj_x + margin_x < ci_x:
                    relations.append((j, i, self.RELATION_TYPES["left_of"]))
                    relations.append((i, j, self.RELATION_TYPES["right_of"]))

                # --- above/below ---
                margin_y = 0.05 * (ci_h + cj_h)
                if ci_y + margin_y < cj_y:
                    relations.append((i, j, self.RELATION_TYPES["above"]))
                    relations.append((j, i, self.RELATION_TYPES["below"]))
                elif cj_y + margin_y < ci_y:
                    relations.append((j, i, self.RELATION_TYPES["above"]))
                    relations.append((i, j, self.RELATION_TYPES["below"]))

                # --- overlaps (IoU) ---
                iou = self._box_iou(nodes[i].box, nodes[j].box)
                if iou > 0.3:
                    relations.append((i, j, self.RELATION_TYPES["overlaps"]))
                    relations.append((j, i, self.RELATION_TYPES["overlaps"]))

                # --- near (거리 기반) ---
                dx = ci_x - cj_x
                dy = ci_y - cj_y
                dist = math.sqrt(dx * dx + dy * dy)
                # 두 박스 평균 너비의 2배 이내면 근접한 것으로 본다
                avg_w = 0.5 * (ci_w + cj_w)
                if dist < 2.0 * avg_w:
                    relations.append((i, j, self.RELATION_TYPES["near"]))
                    relations.append((j, i, self.RELATION_TYPES["near"]))

        return relations

    def update_focus_classes(self, focus_targets: Iterable[str]):
        self.detector.update_focus_classes(list(focus_targets))

    @property
    def ltm_entities(self):
        try:
            return [
                self._serialize_entity_record(rec)
                for rec in sorted(self.ltm.entities.values(), key=lambda r: r.entity_id)
            ]
        except Exception:
            return []

    def _serialize_entity_record(self, rec: EntityRecord):
        return {
            "entityId": rec.entity_id,
            "baseCls": rec.base_cls,
            "lastBox": [float(x) for x in rec.last_box],
            "lastSeenFrame": rec.last_seen_frame,
            "seenFrames": rec.seen_frames[-20:],
            "trackHistory": rec.track_history[-20:],
            "meta": rec.meta,
        }

    def process_frame(
        self,
        frame_bgr,
        frame_idx: int,
        run_grin: bool = False,
        grin_horizon: int = 16,
        max_entities: int = 16,
    ):
        detections = self.detector.detect(frame_bgr)
        self._log(
            event="detect_done",
            frame_idx=frame_idx,
            num_dets=len(detections),
            matched_brain="PPC",
        )
        nodes = self._assign_entities_with_ltm(detections, frame_idx)
        self._log(
            event="track_update",
            frame_idx=frame_idx,
            num_tracks=len(nodes),
            matched_brain="PPC",
        )

        # --- rule-based relations 생성 ---
        relations = self._build_relations_stub(nodes)
        self._log(
            event="relations_built",
            frame_idx=frame_idx,
            num_relations=len(relations),
            matched_brain="PPC",
        )

        sg_frame = SceneGraphFrame(
            frame_idx=frame_idx,
            nodes=nodes,
            relations=relations,
        )

        self.stm.push(sg_frame)

        grin_outputs: Dict[int, Dict] = {}
        if run_grin:
            person_entities = {n.entity_id for n in nodes if n.cls == "person"}
            for eid in person_entities:
                seq = self.stm.get_sequence_for_grin(
                    end_frame=frame_idx,
                    horizon=grin_horizon,
                    stride=1,
                )
                grin_input = build_grin_input(seq, focus_entity_id=eid)
                if grin_input is None:
                    continue

                node_feats = grin_input["node_feats"].to(self.device)
                boxes = grin_input["boxes"].to(self.device)
                t_idx = grin_input["t_idx"].to(self.device)

                out = self.grin(node_feats, boxes, t_idx)
                grin_outputs[eid] = out
            self._log(
                event="grin_run",
                frame_idx=frame_idx,
                focus_entities=len(person_entities),
                horizon=grin_horizon,
                matched_brain="PPC",
            )

        return sg_frame, grin_outputs
