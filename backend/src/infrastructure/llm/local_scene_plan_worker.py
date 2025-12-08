import threading
from queue import Queue
from typing import Iterable, Tuple, Dict, Any, Optional

from infrastructure.language.scene_plan_generator import generate_scene_plan_local, load_local_llm
from infrastructure.logging.pipeline_logger import PipelineLogger


class LocalScenePlanWorker:
    """
    Background worker that satisfies ScenePlanPort using a local LLM.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        attn_impl: str,
        logger: Optional[PipelineLogger] = None,
        max_new_tokens: int = 96,
        max_objects: int = 12,
        max_relations: int = 24,
    ) -> None:
        self.logger = logger
        self.max_new_tokens = max_new_tokens
        self.max_objects = max_objects
        self.max_relations = max_relations

        self.tokenizer, self.model = load_local_llm(
            model_name=model_name,
            device=device,
            attn_impl=attn_impl,
            logger=logger,
        )

        self.task_q: "Queue[tuple[int, Any]]" = Queue()
        self.result_q: "Queue[tuple[int, Dict[str, Any]]]" = Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, frame_idx: int, simple_scene_graph) -> None:
        self.task_q.put((frame_idx, simple_scene_graph))

    def poll_results(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        while not self.result_q.empty():
            yield self.result_q.get_nowait()

    def shutdown(self) -> None:
        self.task_q.put(None)
        self.thread.join(timeout=1.0)

    def _worker(self) -> None:
        while True:
            entry = self.task_q.get()
            if entry is None:
                self.task_q.task_done()
                break

            frame_idx, simple_sg = entry
            try:
                plan = generate_scene_plan_local(
                    tokenizer=self.tokenizer,
                    model=self.model,
                    sg_frame=simple_sg,
                    max_new_tokens=self.max_new_tokens,
                    max_objects=self.max_objects,
                    max_relations=self.max_relations,
                )
                self.result_q.put((frame_idx, plan))
            except Exception as e:
                self.result_q.put((frame_idx, {"error": str(e)}))
                if self.logger:
                    self.logger.log(
                        module="ScenePlanWorker",
                        event="error",
                        frame_idx=frame_idx,
                        level="ERROR",
                        message=str(e),
                    )
            finally:
                self.task_q.task_done()
