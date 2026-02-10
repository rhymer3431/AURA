import threading
from queue import Queue
from typing import Iterable, Tuple, Dict, Any, Optional, List
import torch

from src.infrastructure.language.scene_plan_generator import generate_scene_plan_local, load_local_llm
from src.infrastructure.logging.pipeline_logger import PipelineLogger

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

        print("[LLM Worker] Loading model in main thread...")
        self.tokenizer, self.model = load_local_llm(
            model_name=model_name,
            attn_impl=attn_impl,
            logger=logger,
        )
        
        # Device 정보
        try:
            print("=== LLM DEVICE MAP ===")
            print(self.model.hf_device_map)
            print("=== LLM PARAM DEVICE ===")
            print(next(self.model.parameters()).device)
            print("=== LLM DTYPE ===")
            print(next(self.model.parameters()).dtype)
            
            if hasattr(self.model.config, '_attn_implementation'):
                print("=== ATTENTION IMPL ===")
                print(self.model.config._attn_implementation)
        except Exception as e:
            print("Device check failed:", e)

        # 🔥 CUDA 최적화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.task_q: "Queue[tuple[int, Any]]" = Queue()
        self.result_q: "Queue[tuple[int, Dict[str, Any]]]" = Queue()
        
        # 🔥 Warmup (간단 버전)
        print("[LLM Worker] Warming up model...")
        self._warmup()
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _warmup(self) -> None:
        """첫 추론 오버헤드 제거"""
        try:
            import time
            dummy_prompt = "Output JSON: {}"
            inputs = self.tokenizer(dummy_prompt, return_tensors="pt").to(self.model.device)
            
            start = time.perf_counter()
            with torch.inference_mode():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    use_cache=True,
                )
            warmup_time = time.perf_counter() - start
            print(f"[LLM Worker] ✓ Warmup completed in {warmup_time:.4f}s")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[LLM Worker] ⚠ Warmup failed: {e}")
            # Warmup 실패해도 계속 진행

    def submit(self, frame_idx: int, simple_scene_graph) -> None:
        self.task_q.put((frame_idx, simple_scene_graph))

    def poll_results(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        while not self.result_q.empty():
            yield self.result_q.get_nowait()

    def shutdown(self) -> None:
        self.task_q.put(None)
        self.thread.join(timeout=2.0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _worker(self) -> None:
        """백그라운드 워커"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.model.device)
        
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
                error_msg = f"Frame {frame_idx} failed: {str(e)}"
                print(f"[LLM Worker] ⚠ {error_msg}")
                self.result_q.put((frame_idx, {"error": str(e), "caption": "", "focus_targets": []}))
                
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