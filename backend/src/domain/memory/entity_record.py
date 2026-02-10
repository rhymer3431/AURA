from dataclasses import dataclass, field
import torch
from typing import List, Dict, Optional

@dataclass
class EntityRecord:
    entity_id: int
    base_cls: str
    proto_feat: torch.Tensor
    last_roi_feat: torch.Tensor
    last_face_feat: Optional[torch.Tensor]
    last_box: List[float]
    last_seen_frame: int
    seen_frames: List[int]
    track_history: List[int]
    meta: Dict

    cls_history: List[str] = field(default_factory=list)
    suspect_count: int = 0
    roi_feat_history: List[torch.Tensor] = field(default_factory=list)
