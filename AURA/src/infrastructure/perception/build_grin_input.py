from typing import List, Dict, Optional, Tuple

import torch

from src.domain.perception.scene_graph_frame import SceneGraphFrame


def build_grin_input(
    seq_frames: List[SceneGraphFrame],
    focus_entity_id: int,
) -> Optional[Dict[str, torch.Tensor]]:
    node_feats = []
    boxes = []
    t_idx = []

    for f in seq_frames:
        node = next((n for n in f.nodes if n.entity_id == focus_entity_id), None)
        if node is None:
            continue
        node_feats.append(node.roi_feat)
        boxes.append(node.box)
        t_idx.append(f.frame_idx)

    if not node_feats:
        return None

    node_feats = torch.stack(node_feats, dim=0)
    boxes = torch.tensor(boxes, dtype=torch.float32)
    t_idx = torch.tensor(t_idx, dtype=torch.long)

    return {
        "node_feats": node_feats,
        "boxes": boxes,
        "t_idx": t_idx,
    }
