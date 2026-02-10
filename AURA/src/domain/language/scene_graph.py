from typing import Any, Dict

import torch


def scene_graph_to_struct(
    sg_frame,
    max_objects: int = 12,
    max_relations: int = 24,
):
    """
    SceneGraphTensorFrame → entity_id 중심 JSON 구조로 변환.
    relations 에서 subject/object 는 node index 대신 entity_id 사용.
    """

    # -----------------------------
    # NODES (entity_id 기반)
    # -----------------------------
    boxes = sg_frame.boxes.cpu().tolist()
    scores = sg_frame.scores.cpu().tolist()
    labels = sg_frame.labels
    track_ids = sg_frame.track_ids
    entity_ids = sg_frame.entity_ids

    # score 상위 max_objects 선택
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)[:max_objects]
    
    # 선택된 entity_id 집합 생성
    keep_entity_ids = set(int(entity_ids[idx]) for idx in order)

    objects = []
    for idx in order:
        ent_id = int(entity_ids[idx])
        x1, y1, x2, y2 = boxes[idx]
        objects.append({
            "entity_id": ent_id,
            "label": labels[idx],
            "score": float(scores[idx]),
            "track_id": int(track_ids[idx]),
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)]
        })

    # -----------------------------
    # RELATIONS (entity_id 기반)
    # -----------------------------
    relations = []

    if hasattr(sg_frame, "static_pairs") and sg_frame.static_pairs is not None:
        static_pairs = sg_frame.static_pairs
        static_names = sg_frame.static_rel_names

        for pair, name in zip(static_pairs, static_names):
            # static_pairs는 entity_id를 사용한다고 가정
            s_eid = int(pair[0])
            t_eid = int(pair[1])

            # 두 entity_id 모두 선택된 객체에 포함되는지 확인
            if s_eid in keep_entity_ids and t_eid in keep_entity_ids:
                relations.append({
                    "subject": s_eid,
                    "object": t_eid,
                    "relation": name,
                })

    # truncate relations
    relations = relations[:max_relations]

    return {
        "frame_idx": sg_frame.frame_idx,
        "objects": objects,
        "relations": relations,
    }
    
    