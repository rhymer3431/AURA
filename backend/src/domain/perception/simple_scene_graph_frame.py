from dataclasses import dataclass
from typing import List
import torch
@dataclass
class SimpleSceneGraphFrame:
    """
    LLM 입력용 간단 SceneGraph.
    - boxes: (N, 4) xyxy
    - labels: 객체 클래스 문자열
    - scores: detection confidence
    - track_ids: tracker ID (ByteTrack 등)
    - entity_ids: LTM에서 부여한 entity_id (person만 유효, 나머지는 -1)
    """
    frame_idx: int
    boxes: torch.Tensor
    labels: List[str]
    scores: torch.Tensor
    track_ids: List[int]
    entity_ids: List[int]
    static_pairs: torch.Tensor
    static_rel_names: List[str]
    temporal_pairs: torch.Tensor
    temporal_rel_names: List[str]