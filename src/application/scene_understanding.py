from domain.pipeline.entity.scene_state import SceneState
from domain.pipeline.service.realtime_pipeline import RealtimePipeline


class SceneUnderstandingUseCase:
    """Thin application layer wrapper around the realtime pipeline."""

    def __init__(self, pipeline: RealtimePipeline):
        self.pipeline = pipeline

    def execute(self, state: SceneState) -> SceneState:
        return self.pipeline.run_step(state)
