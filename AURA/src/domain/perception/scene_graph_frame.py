from dataclasses import dataclass
from typing import List, Tuple

from src.domain.perception.entity_node import EntityNode
@dataclass
class SceneGraphFrame:
    """
    한 프레임의 Scene Graph.
    """
    frame_idx: int
    nodes: List[EntityNode]
    relations: List[Tuple[int, int, int]]
