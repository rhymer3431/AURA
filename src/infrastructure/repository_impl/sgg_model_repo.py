from domain.node.entity.node import Node
from domain.sgg.entity.relation import Edge
from domain.sgg.repository.sgg_model_port import SggModelPort


class SggModelRepository(SggModelPort):
    """Adapter wrapper for SGG backends."""

    def __init__(self, adapter: SggModelPort):
        self.adapter = adapter

    def predict(self, nodes: list[Node]) -> list[Edge]:
        return self.adapter.predict(nodes)
