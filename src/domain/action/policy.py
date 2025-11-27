from dataclasses import dataclass
from typing import Optional

from domain.scene_graph.entities import SceneGraph


@dataclass
class PolicyDecision:
    action: str
    reason: Optional[str] = None


class SimplePolicy:
    """
    Extremely lightweight policy: decide high-level action from scene graph.
    """

    def decide(self, scene_graph: SceneGraph) -> PolicyDecision:
        if scene_graph is None or not scene_graph.nodes:
            return PolicyDecision(action="idle", reason="no targets")

        has_threat = any(
            e.relation in ("approaching", "overlapping") for e in scene_graph.edges
        )
        if has_threat:
            return PolicyDecision(action="alert", reason="approaching object detected")

        return PolicyDecision(action="track", reason="objects present")
