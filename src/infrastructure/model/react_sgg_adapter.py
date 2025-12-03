from src.domain.node.entity.entity_node import Node
from domain.sgg.entity.relation import Edge
from domain.sgg.repository.sgg_model_port import SggModelPort


class ReactSggAdapter(SggModelPort):
    def predict(self, nodes: list[Node]) -> list[Edge]:
        raise NotImplementedError("React/SGG adapter not implemented yet.")
