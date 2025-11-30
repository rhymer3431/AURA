from typing import List

from domain.node.entity.node import Node
from domain.sgg.entity.relation import Edge, SceneGraph


class SceneGraphBuilder:
    def __init__(self, relation_predictor):
        self.relation_predictor = relation_predictor

    def build(self, detections: List[Node]) -> SceneGraph:
        sg = SceneGraph(nodes=detections)
        if hasattr(self.relation_predictor, "predict"):
            edges = self.relation_predictor.predict(sg.nodes)
        elif hasattr(self.relation_predictor, "predict_relations"):
            edges = self.relation_predictor.predict_relations(sg.nodes)
        else:
            raise AttributeError(
                "Relation predictor must expose predict() or predict_relations()"
            )
        sg.edges.extend(edges)
        return sg
