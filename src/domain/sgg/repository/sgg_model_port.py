from abc import ABC, abstractmethod
from typing import List

from src.domain.node.entity.entity_node import Node
from domain.sgg.entity.relation import Edge


class SggModelPort(ABC):
    """Port interface for scene-graph generation backends."""

    @abstractmethod
    def predict(self, nodes: List[Node]) -> List[Edge]:
        raise NotImplementedError
