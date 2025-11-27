# src/robotics/domain/scene_graph/entities.py
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Node:
    id: int
    label: str
    bbox: List[float]
    features: Dict = field(default_factory=dict)

@dataclass
class Edge:
    source: int
    target: int
    relation: str
    confidence: float

@dataclass
class SceneGraph:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
