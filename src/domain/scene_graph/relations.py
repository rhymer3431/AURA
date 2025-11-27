# src/robotics/domain/scene_graph/relations.py
from typing import List
from .entities import Node, Edge


class SimpleRelationInfer:
    def predict_relations(self, nodes: List[Node]) -> List[Edge]:
        edges = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                
                n1, n2 = nodes[i], nodes[j]
                r = self._spatial_relation(n1, n2)
                if r:
                    edges.append(
                        Edge(
                            source=n1.id,
                            target=n2.id,
                            relation=r,
                            confidence=1.0,
                        )
                    )
        return edges

    def _spatial_relation(self, a: Node, b: Node) -> str:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox

        center_a = (ax1 + ax2) / 2
        center_b = (bx1 + bx2) / 2

        # Example heuristics
        if abs(center_a - center_b) < 50:
            return "near"

        if center_a < center_b:
            return "left_of"
        else:
            return "right_of"
