# Domain entities for scene graph relations.
from dataclasses import dataclass, field
from typing import List

from src.domain.node.entity.entity_node import Node


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
