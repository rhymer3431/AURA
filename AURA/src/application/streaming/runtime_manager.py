from typing import Iterable, Optional, Tuple, Dict, Any

from src.infrastructure.logging.pipeline_logger import PipelineLogger
from src.infrastructure.perception.perception_service_adapter import PerceptionServiceAdapter
from src.infrastructure.llm.local_scene_plan_worker import LocalScenePlanWorker
from src.infrastructure.streaming.config import StreamServerConfig
from src.domain.streaming.port import ScenePlanPort


class NoopScenePlanner:
    """Fallback planner used when LLM is disabled or unavailable."""

    def submit(self, frame_idx: int, simple_scene_graph) -> None:
        return None

    def poll_results(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        return ()

    def shutdown(self) -> None:
        return None


class RuntimeManager:
    """Starts/stops infra adapters and exposes them to the interface layer."""

    def __init__(self, config: StreamServerConfig):
        self.config = config
        self.logger = PipelineLogger(enabled=config.logging_enabled)
        self.perception: Optional[PerceptionServiceAdapter] = None
        self.scene_planner: Optional[ScenePlanPort] = None

    def start(self) -> None:
        self.perception = PerceptionServiceAdapter(
            yolo_weight=str(self.config.yolo_weight),
            ltm_feat_dim=256,
            device=self.config.llm_device,
            logger=self.logger,
        )
        if not self.config.llm_enabled:
            self.scene_planner = NoopScenePlanner()
            return

        try:
            self.scene_planner = LocalScenePlanWorker(
                model_name=self.config.llm_model_name,
                device=self.config.llm_device,
                attn_impl=self.config.llm_attn_impl,
                logger=self.logger,
            )
        except Exception as exc:
            self.logger.log(
                module="RuntimeManager",
                event="llm_init_failed",
                error=str(exc),
            )
            self.scene_planner = NoopScenePlanner()

    def shutdown(self) -> None:
        if self.scene_planner is not None:
            self.scene_planner.shutdown()
