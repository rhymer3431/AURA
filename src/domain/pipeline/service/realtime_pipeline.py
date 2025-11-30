from typing import Optional

from domain.detect.service.detection_service import DetectionService
from domain.pipeline.entity.scene_state import SceneState
from domain.reason.service.reasoning_service import ReasoningService
from domain.sgg.service.reasoning import SceneGraphReasoner


class RealtimePipeline:
    """
    Simple orchestrator that wires detection, SGG, and reasoning in order.
    """

    def __init__(
        self,
        detector: DetectionService,
        sg_reasoner: Optional[SceneGraphReasoner] = None,
        policy: Optional[ReasoningService] = None,
    ):
        self.detector = detector
        self.sg_reasoner = sg_reasoner
        self.policy = policy

    def run_step(self, state: SceneState) -> SceneState:
        state.detections = self.detector.track(state.raw_frame)

        if self.sg_reasoner and state.has_detections():
            state.scene_graph = self.sg_reasoner.infer_from_detections(
                state.detections
            )

        if self.policy and state.scene_graph:
            state.policy_output = self.policy.decide(state.scene_graph)

        return state
