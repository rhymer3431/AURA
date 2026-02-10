from collections import deque
from typing import List, Optional, Dict, Any, Tuple

from src.domain.perception.scene_graph_frame import SceneGraphFrame
from src.domain.perception.scene_graph_relation import SceneGraphRelation
from src.infrastructure.logging.pipeline_logger import PipelineLogger


class ShortTermGraphMemory:
    def __init__(self, max_frames: int = 60, logger: Optional[PipelineLogger] = None):
        self.buffer: deque[SceneGraphFrame] = deque(maxlen=max_frames)
        self.last_frame: Optional[SceneGraphFrame] = None  # 🔥 이전 SG 보관
        self.logger = logger

    def _log(self, event: str, frame_idx: Optional[int] = None, **payload):
        if self.logger is not None:
            self.logger.log(
                module="ShortTermMemory",
                event=event,
                frame_idx=frame_idx,
                **payload,
            )

    # ------------------------------------------------------------------
    # 🔥 DIFF LOGIC
    # ------------------------------------------------------------------
    def _normalize_relations(
        self, sg_frame: SceneGraphFrame
    ) -> List[SceneGraphRelation]:
        """
        Convert raw relation tuples (subject_idx, object_idx, predicate)
        into SceneGraphRelation objects keyed by persistent entity_ids.
        """
        normalized: List[SceneGraphRelation] = []
        nodes = sg_frame.nodes

        for rel in getattr(sg_frame, "relations", []):
            if isinstance(rel, SceneGraphRelation):
                normalized.append(rel)
                continue

            if (
                not isinstance(rel, tuple)
                or len(rel) < 3
            ):
                continue

            sub_idx, obj_idx, predicate = rel[:3]
            if (
                sub_idx is None
                or obj_idx is None
                or sub_idx < 0
                or obj_idx < 0
                or sub_idx >= len(nodes)
                or obj_idx >= len(nodes)
            ):
                continue

            subject_id = getattr(nodes[sub_idx], "entity_id", None)
            object_id = getattr(nodes[obj_idx], "entity_id", None)
            if subject_id is None or object_id is None:
                continue

            confidence = 1.0
            if len(rel) > 3:
                try:
                    confidence = float(rel[3])
                except (TypeError, ValueError):
                    confidence = 1.0

            normalized.append(
                SceneGraphRelation(
                    subject_id=int(subject_id),
                    predicate=int(predicate),
                    object_id=int(object_id),
                    confidence=confidence,
                )
            )

        return normalized

    def _compute_diff(
        self, prev: SceneGraphFrame, curr: SceneGraphFrame
    ) -> Dict[str, Any]:
        """
        prev SG와 curr SG의 차이를 비교해 변경된 부분만 반환한다.
        """

        # --- Node diff ---
        prev_nodes = {n.entity_id: n for n in prev.nodes}
        curr_nodes = {n.entity_id: n for n in curr.nodes}

        added_nodes = [curr_nodes[n] for n in curr_nodes.keys() - prev_nodes.keys()]
        removed_nodes = list(prev_nodes.keys() - curr_nodes.keys())


        # --- Relation diff ---
        prev_relations = self._normalize_relations(prev)
        curr_relations = self._normalize_relations(curr)

        prev_edges = {
            (r.subject_id, r.predicate, r.object_id): r for r in prev_relations
        }
        curr_edges = {
            (r.subject_id, r.predicate, r.object_id): r for r in curr_relations
        }

        added_edges = [curr_edges[k] for k in curr_edges.keys() - prev_edges.keys()]
        removed_edges = list(prev_edges.keys() - curr_edges.keys())

        diff = {
            "nodes_added": added_nodes,
            "nodes_removed": removed_nodes,
            "edges_added": added_edges,
            "edges_removed": removed_edges,
        }
        return diff

    # ------------------------------------------------------------------
    def push(self, sg_frame: SceneGraphFrame) -> Dict[str, Any]:
        """
        SG Frame을 push하고, diff까지 계산해서 반환한다.
        프론트로는 diff가 empty가 아닐 때만 전송하면 된다.
        """

        diff = None
        if self.last_frame is not None:
            diff = self._compute_diff(self.last_frame, sg_frame)

            # diff 로그
            self._log(
                event="stm_diff",
                frame_idx=sg_frame.frame_idx,
                diff_summary={
                    "nodes_added": len(diff["nodes_added"]),
                    "nodes_removed": len(diff["nodes_removed"]),
                    "edges_added": len(diff["edges_added"]),
                    "edges_removed": len(diff["edges_removed"]),
                },
                matched_brain="Hippocampus",
            )

        # Frame 저장
        self.buffer.append(sg_frame)
        self.last_frame = sg_frame

        # push 이벤트 로그
        self._log(
            event="stm_push",
            frame_idx=sg_frame.frame_idx,
            size=len(self.buffer),
            matched_brain="Hippocampus",
        )

        return diff  # 🔥 외부에서 diff를 받아서 비어 있지 않은 경우만 스트리밍하게 하면 좋음

    # ------------------------------------------------------------------
    # GRIN용 시퀀스
    # ------------------------------------------------------------------
    def get_sequence_for_grin(
        self,
        end_frame: int,
        horizon: int = 16,
        stride: int = 1,
    ) -> List[SceneGraphFrame]:
        candidates = [
            f
            for f in self.buffer
            if end_frame - horizon + 1 <= f.frame_idx <= end_frame
        ]
        candidates = sorted(candidates, key=lambda f: f.frame_idx)
        seq = candidates[::stride]
        if seq:
            self._log(
                event="stm_get_sequence",
                frame_idx=seq[-1].frame_idx,
                horizon=horizon,
                stride=stride,
                length=len(seq),
                matched_brain="Hippocampus",
            )
        return seq
