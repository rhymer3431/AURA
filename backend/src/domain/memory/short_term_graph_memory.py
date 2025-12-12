from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from src.domain.perception.scene_graph_frame import SceneGraphFrame
from src.infrastructure.logging.pipeline_logger import PipelineLogger


@dataclass
class GraphNodeDTO:
    """Lightweight node representation for JSON serialization (id uses the entity_id as a string)."""

    id: str
    label: str
    cls: str
    score: Optional[float] = None
    box: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdgeDTO:
    """Lightweight edge representation for JSON serialization."""

    id: str
    source: str
    target: str
    predicate: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDiff:
    """Changes detected in the scene graph for a single frame, intended for incremental frontend updates."""

    added_nodes: List[GraphNodeDTO] = field(default_factory=list)
    updated_nodes: List[GraphNodeDTO] = field(default_factory=list)
    removed_node_ids: List[str] = field(default_factory=list)

    added_edges: List[GraphEdgeDTO] = field(default_factory=list)
    updated_edges: List[GraphEdgeDTO] = field(default_factory=list)
    removed_edge_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the diff into JSON-serializable primitives."""
        return {
            "addedNodes": [asdict(n) for n in self.added_nodes],
            "updatedNodes": [asdict(n) for n in self.updated_nodes],
            "removedNodeIds": list(self.removed_node_ids),
            "addedEdges": [asdict(e) for e in self.added_edges],
            "updatedEdges": [asdict(e) for e in self.updated_edges],
            "removedEdgeIds": list(self.removed_edge_ids),
        }

    def is_empty(self) -> bool:
        return not (
            self.added_nodes
            or self.updated_nodes
            or self.removed_node_ids
            or self.added_edges
            or self.updated_edges
            or self.removed_edge_ids
        )


@dataclass
class _STMNode:
    entity_id: int
    cls: str
    box: List[float]
    score: float
    track_id: Optional[int]
    last_seen_frame: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dto(self) -> GraphNodeDTO:
        node_id = str(self.entity_id)
        label = f"{self.cls}#{self.entity_id}" if self.cls else node_id
        meta = {"trackId": self.track_id, **self.meta} if self.meta else {"trackId": self.track_id}
        # Remove None entries to keep payload clean
        meta = {k: v for k, v in meta.items() if v is not None}
        return GraphNodeDTO(
            id=node_id,
            label=label,
            cls=self.cls,
            score=self.score,
            box=[float(x) for x in self.box],
            meta=meta,
        )


@dataclass
class _STMEdge:
    subject_id: int
    object_id: int
    predicate: str
    last_seen_frame: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dto(self) -> GraphEdgeDTO:
        edge_id = f"{self.subject_id}:{self.predicate}:{self.object_id}"
        return GraphEdgeDTO(
            id=edge_id,
            source=str(self.subject_id),
            target=str(self.object_id),
            predicate=self.predicate,
            meta=self.meta,
        )


class ShortTermGraphMemory:
    """
    Maintains a short-term scene graph across frames and exposes diffs.

    - Nodes are keyed by persistent entity_id.
    - Edges are keyed by (subject_entity_id, predicate, object_entity_id).
    - Node ids are stringified entity_ids; edge ids use "subject:predicate:object".
    - update_from_frame mutates the STM to match the latest frame, emits a GraphDiff,
      and evicts stale nodes/edges when they fall outside max_age.
    """

    def __init__(
        self,
        max_frames: int = 60,
        logger: Optional[PipelineLogger] = None,
        relation_id_to_name: Optional[Dict[int, str]] = None,
        max_age: Optional[int] = None,
    ):
        self.buffer: deque[SceneGraphFrame] = deque(maxlen=max_frames)
        self.logger = logger
        self.max_age = max_age if max_age is not None else max_frames
        self._nodes: Dict[int, _STMNode] = {}
        self._edges: Dict[Tuple[int, str, int], _STMEdge] = {}
        self._relation_id_to_name = relation_id_to_name or {}

    def _log(self, event: str, frame_idx: Optional[int] = None, **payload):
        if self.logger is not None:
            self.logger.log(
                module="ShortTermMemory",
                event=event,
                frame_idx=frame_idx,
                **payload,
            )

    def _relation_label(self, rel_id: Any) -> str:
        """Return a human-readable predicate string for the relation id."""
        return self._relation_id_to_name.get(rel_id, str(rel_id))

    def _gc(self, frame_idx: int, diff: GraphDiff):
        """Remove nodes/edges that have not been seen for max_age frames."""
        stale_nodes = [
            nid
            for nid, node in self._nodes.items()
            if frame_idx - node.last_seen_frame >= self.max_age
        ]
        for nid in stale_nodes:
            self._nodes.pop(nid, None)
            diff.removed_node_ids.append(str(nid))

        stale_edges = [
            key
            for key, edge in self._edges.items()
            if frame_idx - edge.last_seen_frame >= self.max_age
            or key[0] not in self._nodes
            or key[2] not in self._nodes
        ]
        for key in stale_edges:
            self._edges.pop(key, None)
            edge_id = f"{key[0]}:{key[1]}:{key[2]}"
            diff.removed_edge_ids.append(edge_id)

    def update_from_frame(self, sg_frame: SceneGraphFrame) -> GraphDiff:
        """
        Update STM using the given SceneGraphFrame and return a GraphDiff.

        A node/edge is considered updated if any tracked attributes change OR
        if its last_seen_frame advances (i.e., the object/relation is observed again).
        A node/edge is considered removed if it disappears from the current frame or
        is evicted by GC (max_age).

        The returned GraphDiff can be serialized via .to_dict() for frontend consumption.
        """
        prev_node_ids = set(self._nodes.keys())
        prev_edge_keys = set(self._edges.keys())
        current_node_ids: set[int] = set()
        current_edge_keys: set[Tuple[int, str, int]] = set()

        diff = GraphDiff()

        # Maintain frame buffer for GRIN use-cases.
        self.buffer.append(sg_frame)
        self._log(
            event="stm_push",
            frame_idx=sg_frame.frame_idx,
            size=len(self.buffer),
            matched_brain="Hippocampus",
        )

        # --- nodes ---
        for node in sg_frame.nodes:
            entity_id = int(node.entity_id)
            cls = node.cls
            box = [float(x) for x in node.box]
            score = float(getattr(node, "score", 1.0))
            track_id = node.track_id if node.track_id is not None else None
            meta = {"trackId": track_id, "lastSeenFrame": sg_frame.frame_idx}

            existing = self._nodes.get(entity_id)
            if existing is None:
                new_node = _STMNode(
                    entity_id=entity_id,
                    cls=cls,
                    box=box,
                    score=score,
                    track_id=track_id,
                    last_seen_frame=sg_frame.frame_idx,
                    meta=meta,
                )
                self._nodes[entity_id] = new_node
                diff.added_nodes.append(new_node.to_dto())
            else:
                changed = (
                    existing.cls != cls
                    or existing.box != box
                    or existing.score != score
                    or existing.track_id != track_id
                    or existing.last_seen_frame != sg_frame.frame_idx
                    or existing.meta != meta
                )
                existing.cls = cls
                existing.box = box
                existing.score = score
                existing.track_id = track_id
                existing.last_seen_frame = sg_frame.frame_idx
                existing.meta = meta
                if changed:
                    diff.updated_nodes.append(existing.to_dto())
            current_node_ids.add(entity_id)

        removed_nodes = prev_node_ids - current_node_ids
        for nid in removed_nodes:
            self._nodes.pop(nid, None)
            diff.removed_node_ids.append(str(nid))

        # --- edges ---
        for rel in getattr(sg_frame, "relations", []):
            if len(rel) != 3:
                continue
            sub_idx, obj_idx, rel_id = rel
            if sub_idx >= len(sg_frame.nodes) or obj_idx >= len(sg_frame.nodes):
                continue
            subj_node = sg_frame.nodes[sub_idx]
            obj_node = sg_frame.nodes[obj_idx]
            subj_id = int(subj_node.entity_id)
            obj_id = int(obj_node.entity_id)
            predicate = self._relation_label(rel_id)
            edge_key = (subj_id, predicate, obj_id)
            meta = {"relationId": int(rel_id), "lastSeenFrame": sg_frame.frame_idx}

            existing_edge = self._edges.get(edge_key)
            if existing_edge is None:
                new_edge = _STMEdge(
                    subject_id=subj_id,
                    object_id=obj_id,
                    predicate=predicate,
                    last_seen_frame=sg_frame.frame_idx,
                    meta=meta,
                )
                self._edges[edge_key] = new_edge
                diff.added_edges.append(new_edge.to_dto())
            else:
                changed = (
                    existing_edge.last_seen_frame != sg_frame.frame_idx
                    or existing_edge.meta != meta
                    or existing_edge.predicate != predicate
                )
                existing_edge.last_seen_frame = sg_frame.frame_idx
                existing_edge.meta = meta
                existing_edge.predicate = predicate
                if changed:
                    diff.updated_edges.append(existing_edge.to_dto())
            current_edge_keys.add(edge_key)

        removed_edges = prev_edge_keys - current_edge_keys
        for key in removed_edges:
            self._edges.pop(key, None)
            edge_id = f"{key[0]}:{key[1]}:{key[2]}"
            diff.removed_edge_ids.append(edge_id)

        # Clean up stale nodes/edges based on max_age
        self._gc(frame_idx=sg_frame.frame_idx, diff=diff)

        return diff

    def push(self, sg_frame: SceneGraphFrame) -> GraphDiff:
        """
        Backward-compatible alias to update_from_frame.
        """
        return self.update_from_frame(sg_frame)

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

    def to_graph_data(self) -> Dict[str, Any]:
        """
        Return the full active graph as a dict suitable for the frontend.
        """
        nodes = [n.to_dto() for n in self._nodes.values()]
        edges = [e.to_dto() for e in self._edges.values()]
        return {
            "nodes": [asdict(n) for n in nodes],
            "edges": [asdict(e) for e in edges],
        }
