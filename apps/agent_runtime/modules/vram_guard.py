from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Awaitable, Callable, Dict, List, Optional


class VRAMGuard:
    """Monitors GPU memory and emits degrade levels."""

    def __init__(self, cfg: Dict) -> None:
        self.check_interval_s = float(cfg.get("check_interval_s", 2.0))
        self.safety_threshold_mb = int(cfg.get("safety_threshold_mb", 2900))
        self.device_index = int(cfg.get("device_index", 0))
        self.simulate_if_unavailable = bool(cfg.get("simulate_if_unavailable", True))

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[int, int, int], Awaitable[None] | None]] = []
        self._degrade_level = 0
        self._nvml = None
        self._nvml_handle = None
        self._last_log = 0.0

        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            logging.info("NVML initialized for GPU index %s", self.device_index)
        except Exception as exc:
            logging.warning("NVML unavailable. VRAM guard simulation mode: %s", exc)
            self._nvml = None
            self._nvml_handle = None

    def register_callback(
        self, callback: Callable[[int, int, int], Awaitable[None] | None]
    ) -> None:
        self._callbacks.append(callback)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="vram_guard")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None

    def _query_memory_mb(self) -> tuple[int, int]:
        if self._nvml is not None and self._nvml_handle is not None:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            free_mb = int(info.free / (1024 * 1024))
            used_mb = int(info.used / (1024 * 1024))
            return free_mb, used_mb

        if self.simulate_if_unavailable:
            phase = int(time.time()) % 50
            if phase < 10:
                return 5200, 8200
            if phase < 18:
                return 2600, 10800
            if phase < 25:
                return 1800, 11600
            return 6400, 7000
        return 9999, 0

    async def _run_loop(self) -> None:
        while self._running:
            free_mb, used_mb = self._query_memory_mb()
            level = self._compute_degrade_level(free_mb)

            if level != self._degrade_level:
                self._degrade_level = level
                logging.warning(
                    "VRAM guard level changed: level=%s free=%sMB used=%sMB",
                    level,
                    free_mb,
                    used_mb,
                )
                for callback in self._callbacks:
                    out = callback(level, free_mb, used_mb)
                    if inspect.isawaitable(out):
                        await out
            elif time.time() - self._last_log > 8.0:
                logging.info("VRAM status: free=%sMB used=%sMB level=%s", free_mb, used_mb, level)
                self._last_log = time.time()

            await asyncio.sleep(self.check_interval_s)

    def _compute_degrade_level(self, free_mb: int) -> int:
        if free_mb < int(self.safety_threshold_mb * 0.55):
            return 3
        if free_mb < int(self.safety_threshold_mb * 0.75):
            return 2
        if free_mb < self.safety_threshold_mb:
            return 1
        return 0

