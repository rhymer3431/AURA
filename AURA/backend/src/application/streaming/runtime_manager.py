from typing import Optional

from infrastructure.logging.pipeline_logger import PipelineLogger
from infrastructure.perception.perception_service_adapter import PerceptionServiceAdapter
from infrastructure.llm.local_scene_plan_worker import LocalScenePlanWorker
from infrastructure.streaming.config import StreamServerConfig


class RuntimeManager:
    """Starts/stops infra adapters and exposes them to the interface layer."""

    def __init__(self, config: StreamServerConfig):
        self.config = config
        self.logger = PipelineLogger(enabled=config.logging_enabled)
        self.perception: Optional[PerceptionServiceAdapter] = None
        self.scene_planner: Optional[LocalScenePlanWorker] = None

    def start(self) -> None:
        self.perception = PerceptionServiceAdapter(
            yolo_weight=str(self.config.root_dir / "yolov8s-worldv2.pt"),
            ltm_feat_dim=256,
            device=self.config.llm_device,
            logger=self.logger,
        )
        self.scene_planner = LocalScenePlanWorker(
            model_name=self.config.llm_model_name,
            device=self.config.llm_device,
            attn_impl=self.config.llm_attn_impl,
            logger=self.logger,
        )

    def shutdown(self) -> None:
        if self.scene_planner is not None:
            self.scene_planner.shutdown()
