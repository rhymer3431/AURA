from typing import List

from src.domain.node.entity.entity_node import Node
from domain.sgg.entity.relation import Edge


class SimpleRelationInfer:
    """
    Lightweight heuristic relation predictor used as a default SGG backend.
    """

    def predict_relations(self, nodes: List[Node]) -> List[Edge]:
        edges = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue

                n1, n2 = nodes[i], nodes[j]
                relation = self._spatial_relation(n1, n2)
                if relation:
                    edges.append(
                        Edge(
                            source=n1.id,
                            target=n2.id,
                            relation=relation,
                            confidence=1.0,
                        )
                    )
        return edges

    def _spatial_relation(self, a: Node, b: Node) -> str:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox

        center_a = (ax1 + ax2) / 2
        center_b = (bx1 + bx2) / 2

        if abs(center_a - center_b) < 50:
            return "near"

        if center_a < center_b:
            return "left_of"
        return "right_of"
