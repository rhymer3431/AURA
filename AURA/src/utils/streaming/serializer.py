from typing import Dict, Any, Optional, List
from src.domain.memory.entity_record import EntityRecord
from src.domain.perception.entity_node import EntityNode
from src.domain.perception.scene_graph_relation import SceneGraphRelation


# ============================================================
# 🔹 EntityRecord 직렬화
# ============================================================
def serialize_entity_record(rec: EntityRecord) -> Dict[str, Any]:
    return {
        "entityId": rec.entity_id,
        "baseCls": rec.base_cls,
        "lastBox": [float(x) for x in rec.last_box],
        "lastSeenFrame": rec.last_seen_frame,
        "seenCount": int(getattr(rec, "seen_count", 0) or 0),
        "trackHistory": rec.track_history[-20:],
        "meta": rec.meta,
    }


# ============================================================
# 🔹 Scene Graph Node 직렬화
# ============================================================
def serialize_node(node: EntityNode) -> Dict[str, Any]:
    return {
        "entityId": int(node.entity_id),
        "trackId": node.track_id if node.track_id is not None else -1,
        "cls": node.cls,
        "box": [float(x) for x in node.box],
        "score": float(getattr(node, "score", 1.0)),
    }


def serialize_frame_entities(sg_frame) -> List[Dict[str, Any]]:
    return [serialize_node(n) for n in sg_frame.nodes]


# ============================================================
# 🔹 Scene Graph Relation 직렬화
# ============================================================
def serialize_relation(rel: SceneGraphRelation,
                       rel_id_to_name: Dict[int, str],
                       subj: EntityNode,
                       obj: EntityNode) -> Dict[str, Any]:

    predicate = getattr(rel, "predicate", getattr(rel, "relation_id", None))
    return {
        "subjectEntityId": int(subj.entity_id),
        "objectEntityId": int(obj.entity_id),
        "relation": rel_id_to_name.get(predicate, str(predicate)),
        "relationId": int(predicate),
        "confidence": float(getattr(rel, "confidence", 1.0)),
    }


def serialize_relations(sg_frame, rel_id_to_name: Dict[int, str]) -> List[Dict[str, Any]]:
    relations: List[Dict[str, Any]] = []
    nodes = sg_frame.nodes

    for raw_rel in getattr(sg_frame, "relations", []):
        if isinstance(raw_rel, SceneGraphRelation):
            subj = next((n for n in nodes if n.entity_id == raw_rel.subject_id), None)
            obj = next((n for n in nodes if n.entity_id == raw_rel.object_id), None)
            if subj is None or obj is None:
                continue
            rel_obj = raw_rel
        else:
            if not isinstance(raw_rel, tuple) or len(raw_rel) < 3:
                continue
            sub_idx, obj_idx, rel_id = raw_rel[:3]
            if sub_idx >= len(nodes) or obj_idx >= len(nodes):
                continue
            subj = nodes[sub_idx]
            obj = nodes[obj_idx]
            confidence = 1.0
            if len(raw_rel) > 3:
                try:
                    confidence = float(raw_rel[3])
                except (TypeError, ValueError):
                    confidence = 1.0
            rel_obj = SceneGraphRelation(subj.entity_id, rel_id, obj.entity_id, confidence)

        relations.append(serialize_relation(rel_obj, rel_id_to_name, subj, obj))

    return relations


# ============================================================
# 🔹 SG Diff 직렬화기 (🔥 새로 추가된 핵심 부분)
# ============================================================

def _serialize_edge_key(edge_key: tuple,
                        relation_name_map: Dict[int, str]) -> Dict[str, Any]:
    """Convert removed edge key (subject, predicate, object) into JSON."""
    subject_id, predicate, object_id = edge_key

    return {
        "subject": int(subject_id),
        "predicate": relation_name_map.get(predicate, str(predicate)),
        "object": int(object_id),
    }


def serialize_sg_diff(
    sg_diff: Dict[str, Any],
    relation_name_map: Dict[int, str],
) -> Dict[str, Any]:
    """
    Convert raw SG diff (nodes_added, edges_removed 등) into JSON-serializable format.
    """
    return {
        "nodesAdded": [
            serialize_node(n) for n in sg_diff.get("nodes_added", [])
        ],

        "nodesRemoved": [
            int(n) for n in sg_diff.get("nodes_removed", [])
        ],
        "edgesAdded": [
            {
                "subject": int(e.subject_id),
                "predicate": relation_name_map.get(e.predicate, str(e.predicate)),
                "object": int(e.object_id),
                "confidence": float(getattr(e, "confidence", 1.0)),
            }
            for e in sg_diff.get("edges_added", [])
        ],

        "edgesRemoved": [
            _serialize_edge_key(e, relation_name_map)
            for e in sg_diff.get("edges_removed", [])
        ],

        "edgesUpdated": [
            {
                "subject": int(e.subject_id),
                "predicate": relation_name_map.get(e.predicate, str(e.predicate)),
                "object": int(e.object_id),
                "confidence": float(getattr(e, "confidence", 1.0)),
            }
            for e in sg_diff.get("edges_updated", [])
        ],
    }
