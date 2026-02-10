from dataclasses import dataclass, field
import torch
from typing import List, Dict, Optional

@dataclass
class EntityRecord:
    entity_id: int
    base_cls: str

    # --- identity ---
    proto_feat: torch.Tensor            # EMA 기반 대표 임베딩 (핵심)
    last_roi_feat: torch.Tensor
    last_face_feat: Optional[torch.Tensor]

    # --- spatial / temporal ---
    last_box: List[float]
    last_seen_frame: int                # 마지막 관측 시점
    seen_count: int                     # 총 관측 횟수 (누적)

    # --- tracking ---
    track_history: List[int]

    # --- consistency / metadata ---
    cls_history: List[str] = field(default_factory=list)
    suspect_count: int = 0
    roi_feat_history: List[torch.Tensor] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)
