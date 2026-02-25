from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

from .contracts import Detection2D3D


class YOLOEPerception:
    """YOLOE-26 TRT wrapper with queue-drop behavior and mock fallback."""

    def __init__(
        self,
        cfg: Dict,
        on_detections: Optional[Callable[[List[Detection2D3D]], Awaitable[None] | None]] = None,
    ) -> None:
        self.engine_path = str(cfg.get("engine_path", "models/yoloe_26m_fp8.engine"))
        self.queue_size = int(cfg.get("queue_size", 2))
        self.base_interval_s = float(cfg.get("base_interval_s", 0.50))
        self.warmup_iters = int(cfg.get("warmup_iters", 2))
        self.mock_mode = bool(cfg.get("mock_mode", True))
        self.fallback_to_mock = bool(cfg.get("fallback_to_mock", True))
        self.enable_fallback_variant = bool(cfg.get("enable_fallback_variant", True))
        self.on_detections = on_detections

        self._frame_queue: Deque[Any] = deque(maxlen=max(1, self.queue_size))
        self._drop_count = 0
        self._tick = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._degrade_level = 0
        self._class_cycle = ["apple", "bottle", "cup", "book"]
        self._active_variant = "M"

    async def warmup(self) -> None:
        if not self.mock_mode:
            if not Path(self.engine_path).exists():
                if self.fallback_to_mock:
                    logging.warning(
                        "YOLOE engine not found (%s). Switching to mock mode.", self.engine_path
                    )
                    self.mock_mode = True
                else:
                    raise FileNotFoundError(
                        f"YOLOE engine not found and fallback_to_mock=false: {self.engine_path}"
                    )
            else:
                logging.info("Loading YOLOE TensorRT engine: %s", self.engine_path)
                # TODO: Real TensorRT engine load and context allocation.
        else:
            logging.info("YOLOE mock mode enabled.")

        for i in range(self.warmup_iters):
            await asyncio.sleep(0.15)
            logging.info("YOLOE warmup iteration %s/%s", i + 1, self.warmup_iters)

    def submit_frame(self, frame: Any) -> None:
        if len(self._frame_queue) >= self._frame_queue.maxlen:
            self._drop_count += 1
        self._frame_queue.append(frame)

    def set_degrade_level(self, level: int) -> None:
        self._degrade_level = max(0, int(level))
        if self._degrade_level >= 2 and self.enable_fallback_variant:
            self._active_variant = "S"
        else:
            self._active_variant = "M"
        logging.info(
            "YOLOE degrade level=%s, active_variant=%s, dropped_frames=%s",
            self._degrade_level,
            self._active_variant,
            self._drop_count,
        )

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="perception_yoloe")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            detections = self._infer_once()
            if detections and self.on_detections is not None:
                out = self.on_detections(detections)
                if inspect.isawaitable(out):
                    await out
            await asyncio.sleep(self._effective_interval_s())

    def _effective_interval_s(self) -> float:
        if self._degrade_level <= 0:
            return self.base_interval_s
        if self._degrade_level == 1:
            return self.base_interval_s * 1.5
        if self._degrade_level == 2:
            return self.base_interval_s * 2.5
        return self.base_interval_s * 3.5

    def _infer_once(self) -> List[Detection2D3D]:
        self._tick += 1
        if not self.mock_mode:
            # TODO: Execute TensorRT inference from latest frame and decode results.
            return []

        # In mock mode, synthesize detections while preserving the same output interface.
        if self._tick % 2 != 0:
            return []

        cls_name = self._class_cycle[(self._tick // 2) % len(self._class_cycle)]
        score = 0.75 + random.random() * 0.2
        detection = Detection2D3D(
            object_id=f"{cls_name}-0",
            class_name=cls_name,
            score=min(0.99, score),
            bbox_xywh=(220.0, 130.0, 85.0, 95.0),
            position_in_map=None,
            timestamp=time.time(),
        )
        logging.info(
            "Perception detection: class=%s score=%.2f queue_drop=%s variant=%s",
            detection.class_name,
            detection.score,
            self._drop_count,
            self._active_variant,
        )
        return [detection]
