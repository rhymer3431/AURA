from typing import Any, Dict

import torch


def scene_graph_to_struct(
    sg_frame,
    max_objects: int = 12,
    max_relations: int = 24,
) -> Dict[str, Any]:
    """
    Convert a SceneGraphFrame into a compact dict for LLM prompts.
    Downsamples objects/relations to keep prompts short.
    """
    boxes = sg_frame.boxes
    scores = sg_frame.scores
    track_ids = sg_frame.track_ids
    labels = sg_frame.labels
    entity_ids = getattr(sg_frame, "entity_ids", None)

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().tolist()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().tolist()
    if isinstance(track_ids, torch.Tensor):
        track_ids = track_ids.cpu().tolist()
    if entity_ids is None:
        entity_ids = [-1] * len(boxes)
    elif isinstance(entity_ids, torch.Tensor):
        entity_ids = entity_ids.cpu().tolist()

    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)[:max_objects]
    keep_set = set(order)

    objects = []
    old_to_new = {}
    for new_idx, i in enumerate(order):
        x1, y1, x2, y2 = boxes[i]
        old_to_new[i] = new_idx
        objects.append(
            {
                "index": new_idx,
                "orig_index": int(i),
                "track_id": int(track_ids[i]),
                "entity_id": int(entity_ids[i]),
                "label": labels[i],
                "score": float(scores[i]),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            }
        )

    relations = []
    if hasattr(sg_frame, "static_pairs"):
        static_pairs = getattr(sg_frame, "static_pairs", [])
        static_names = getattr(sg_frame, "static_rel_names", [])
        if static_pairs is not None and static_names is not None:
            for (s, t), name in zip(static_pairs, static_names):
                if s in keep_set and t in keep_set:
                    relations.append(
                        {
                            "subject": old_to_new[int(s)],
                            "object": old_to_new[int(t)],
                            "relation": name,
                        }
                    )

    return {
        "frame_idx": sg_frame.frame_idx,
        "objects": objects,
        "relations": relations[:max_relations],
    }
