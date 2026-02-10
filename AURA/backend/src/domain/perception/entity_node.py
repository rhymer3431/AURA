from dataclasses import dataclass
import torch
from typing import List, Optional
@dataclass
class EntityNode:
    """
    LTM에서 부여한 entity_id를 포함하는 노드 구조.
    """
    entity_id: int
    track_id: Optional[int]
    cls: str
    box: List[float]
    roi_feat: torch.Tensor
    frame_idx: int
    score: float = 1.0  # detection score 보존용 (기본값 1.0)
