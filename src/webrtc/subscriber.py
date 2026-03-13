from __future__ import annotations

import asyncio
from contextlib import suppress
import time

import numpy as np

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter
from ipc.base import MessageBus
from ipc.messages import ActionCommand
from ipc.shm_ring import SharedMemoryRing
from ipc.zmq_bus import ZmqBus

from .config import WebRTCGatewayConfig
from .models import (
    FrameCache,
    GatewayEvent,
    build_frame_meta_message,
    build_snapshot_message,
    build_waiting_for_frame_message,
    frame_age_ms,
    ipc_message_event,
    is_frame_stale,
)


class ObservationSubscriber:
    def __init__(
        self,
        config: WebRTCGatewayConfig,
        *,
        bus: MessageBus | None = None,
        shm_ring: SharedMemoryRing | None = None,
    ) -> None:
        self.config = config
        self._bus = bus or ZmqBus(
            control_endpoint=str(config.control_endpoint),
            telemetry_endpoint=str(config.telemetry_endpoint),
            role="agent",
            identity=str(config.identity),
        )
        self._owns_bus = bus is None
        self._shm_ring = shm_ring
        self._owns_shm = shm_ring is None
        self._adapter = IsaacBridgeAdapter(self._bus, shm_ring=self._shm_ring)
        self._task: asyncio.Task[None] | None = None
        self._frame: FrameCache | None = None
        self._seq = 0
        self._listeners: list[asyncio.Queue[GatewayEvent]] = []
        self._latest_command: ActionCommand | None = None

    @property
    def current_frame(self) -> FrameCache | None:
        return self._frame

    @property
    def latest_command_type(self) -> str:
        if self._latest_command is None:
            return ""
        return str(self._latest_command.action_type)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop(), name="webrtc-observation-subscriber")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._owns_shm and self._shm_ring is not None:
            self._shm_ring.close()
            self._shm_ring = None
        if self._owns_bus:
            self._bus.close()

    def add_listener(self, *, maxsize: int = 64) -> asyncio.Queue[GatewayEvent]:
        queue: asyncio.Queue[GatewayEvent] = asyncio.Queue(maxsize=max(int(maxsize), 1))
        self._listeners.append(queue)
        return queue

    def remove_listener(self, queue: asyncio.Queue[GatewayEvent]) -> None:
        self._listeners = [item for item in self._listeners if item is not queue]

    def last_frame_age_ms(self) -> float | None:
        return frame_age_ms(self._frame)

    def build_state_snapshot(self) -> dict[str, object]:
        if self._frame is None or is_frame_stale(self._frame, stale_after_sec=self.config.stale_frame_timeout_sec):
            return build_waiting_for_frame_message(
                age_ms=frame_age_ms(self._frame),
                has_seen_frame=self._frame is not None,
            )
        return build_snapshot_message(self._frame, active_command_type=self.latest_command_type)

    def build_frame_meta(self) -> dict[str, object] | None:
        if self._frame is None or is_frame_stale(self._frame, stale_after_sec=self.config.stale_frame_timeout_sec):
            return None
        return build_frame_meta_message(self._frame)

    async def _poll_loop(self) -> None:
        sleep_interval = max(float(self.config.poll_interval_ms), 1.0) / 1000.0
        while True:
            processed = 0
            self._attach_shm_if_needed()

            for command in self._adapter.drain_commands():
                self._latest_command = command
                processed += 1

            for status in self._adapter.drain_statuses():
                self._publish_event(GatewayEvent(kind="status", payload=ipc_message_event("status", status)))
                processed += 1

            for notice in self._adapter.drain_notices():
                self._publish_event(GatewayEvent(kind="notice", payload=ipc_message_event("notice", notice)))
                processed += 1

            for capability in self._adapter.drain_capabilities():
                self._publish_event(GatewayEvent(kind="capability", payload=ipc_message_event("capability", capability)))
                processed += 1

            for ping in self._adapter.drain_health():
                self._publish_event(GatewayEvent(kind="health", payload=ipc_message_event("health", ping)))
                processed += 1

            for frame_header in self._adapter.drain_frame_headers():
                processed += 1
                try:
                    batch = self._adapter.reconstruct_batch(frame_header)
                except FileNotFoundError:
                    self._reset_shm()
                    continue
                except RuntimeError:
                    continue
                if batch.rgb_image is None:
                    continue
                overlay = frame_header.metadata.get("viewer_overlay", {})
                if not isinstance(overlay, dict):
                    overlay = {}
                self._seq += 1
                self._frame = FrameCache(
                    seq=int(self._seq),
                    frame_header=frame_header,
                    rgb_image=np.asarray(batch.rgb_image, dtype=np.uint8),
                    depth_image_m=None if batch.depth_image_m is None else np.asarray(batch.depth_image_m, dtype=np.float32),
                    viewer_overlay=dict(overlay),
                    last_frame_monotonic=time.monotonic(),
                )

            await asyncio.sleep(0.0 if processed > 0 else sleep_interval)

    def _attach_shm_if_needed(self) -> None:
        if self._shm_ring is not None or not self._owns_shm:
            return
        try:
            self._shm_ring = SharedMemoryRing(
                name=str(self.config.shm_name),
                slot_size=int(self.config.shm_slot_size),
                capacity=int(self.config.shm_capacity),
                create=False,
            )
        except FileNotFoundError:
            return
        self._adapter = IsaacBridgeAdapter(self._bus, shm_ring=self._shm_ring)

    def _reset_shm(self) -> None:
        if self._owns_shm and self._shm_ring is not None:
            with suppress(Exception):
                self._shm_ring.close()
        self._shm_ring = None
        self._adapter = IsaacBridgeAdapter(self._bus, shm_ring=None)

    def _publish_event(self, event: GatewayEvent) -> None:
        alive: list[asyncio.Queue[GatewayEvent]] = []
        for queue in self._listeners:
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue
            alive.append(queue)
        self._listeners = alive
