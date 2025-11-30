# Domain entity representing a single object/node in the scene.
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Node:
    id: int
    label: str
    bbox: List[float]
    features: Dict = field(default_factory=dict)
