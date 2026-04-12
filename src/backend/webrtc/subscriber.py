"""Subscribe to live viewer frames published by control_runtime."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import time

import numpy as np

from systems.shared.viewer_transport import VIEWER_HEALTH_TOPIC, VIEWER_OBSERVATION_TOPIC
from systems.transport import SharedMemoryRing, ZmqBus, decode_ndarray, ref_from_dict

from .config import WebRTCServiceConfig
from .models import (
    FrameCache,
    build_frame_meta_message,
    build_frame_state_message,
    build_waiting_for_frame_message,
    frame_age_ms,
    is_frame_stale,
)


_SHM_OVERWRITE_ERROR = "Shared memory slot was overwritten before it was read."


def _overlay_value(metadata: dict[str, object], *keys: str) -> object:
    for key in keys:
        if key in metadata:
            return metadata.get(key)
    return None


def _extract_viewer_overlay(metadata: dict[str, object]) -> dict[str, object]:
    overlay = metadata.get("viewer_overlay", {})
    if not isinstance(overlay, dict):
        overlay = {}
    normalized = dict(overlay)
    trajectory_pixels = _overlay_value(metadata, "trajectory_pixels", "trajectoryPixels")
    if isinstance(trajectory_pixels, list):
        normalized["trajectory_pixels"] = trajectory_pixels
        normalized["trajectoryPixels"] = trajectory_pixels
    system2_pixel_goal = _overlay_value(metadata, "system2_pixel_goal", "system2PixelGoal")
    if isinstance(system2_pixel_goal, list) and len(system2_pixel_goal) >= 2:
        normalized["system2_pixel_goal"] = system2_pixel_goal
        normalized["system2PixelGoal"] = system2_pixel_goal
    active_target = _overlay_value(metadata, "active_target", "activeTarget")
    if isinstance(active_target, dict) and active_target:
        normalized["active_target"] = dict(active_target)
        normalized["activeTarget"] = dict(active_target)
    detections = metadata.get("detections")
    if isinstance(detections, list):
        normalized["detections"] = detections
    detector_backend = metadata.get("detector_backend")
    if isinstance(detector_backend, str) and detector_backend:
        normalized["detector_backend"] = detector_backend
    return normalized


def _is_transient_decode_error(exc: RuntimeError) -> bool:
    return str(exc) == _SHM_OVERWRITE_ERROR


class ObservationSubscriber:
    def __init__(
        self,
        config: WebRTCServiceConfig,
        *,
        bus: ZmqBus | None = None,
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
        self._task: asyncio.Task[None] | None = None
        self._frame: FrameCache | None = None
        self._seq = 0
        self._latest_health: dict[str, object] = {}
        self._stream_stalled = False
        self._decode_ok = 0
        self._decode_drops = 0
        self._shm_overwrite_drops = 0
        self._stale_transitions = 0

    @property
    def current_frame(self) -> FrameCache | None:
        return self._frame

    @property
    def latest_health(self) -> dict[str, object]:
        return dict(self._latest_health)

    @property
    def debug_counters(self) -> dict[str, int]:
        return {
            "decodeOk": int(self._decode_ok),
            "decodeDrops": int(self._decode_drops),
            "shmOverwriteDrops": int(self._shm_overwrite_drops),
            "staleTransitions": int(self._stale_transitions),
        }

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop(), name="backend-webrtc-subscriber")

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

    def last_frame_age_ms(self) -> float | None:
        return frame_age_ms(self._frame)

    def build_state_snapshot(self) -> dict[str, object]:
        frame = self._frame
        if frame is None:
            return build_waiting_for_frame_message(age_ms=frame_age_ms(self._frame), has_seen_frame=self._frame is not None)
        payload = build_frame_state_message(frame)
        payload.update(self._stream_fields(frame))
        return payload

    def build_frame_meta(self) -> dict[str, object] | None:
        frame = self._frame
        if frame is None:
            return None
        payload = build_frame_meta_message(frame)
        payload.update(self._stream_fields(frame))
        return payload

    async def _poll_loop(self) -> None:
        sleep_interval = max(float(self.config.poll_interval_ms), 1.0) / 1000.0
        while True:
            processed = 0
            self._attach_shm_if_needed()

            for record in self._bus.poll(VIEWER_HEALTH_TOPIC, max_items=8):
                details = getattr(record.message, "details", {})
                if isinstance(details, dict):
                    self._latest_health = dict(details.get("viewer", {})) if isinstance(details.get("viewer", {}), dict) else {}
                processed += 1

            for record in self._bus.poll(VIEWER_OBSERVATION_TOPIC, max_items=32):
                processed += 1
                header = record.message
                metadata = dict(header.metadata)
                try:
                    rgb = self._decode_rgb(metadata)
                    depth = self._decode_depth(metadata)
                except FileNotFoundError:
                    self._decode_drops += 1
                    self._reset_shm()
                    continue
                except RuntimeError as exc:
                    if _is_transient_decode_error(exc):
                        self._decode_drops += 1
                        self._shm_overwrite_drops += 1
                        continue
                    raise
                if rgb is None:
                    self._decode_drops += 1
                    continue
                overlay = _extract_viewer_overlay(metadata)
                self._seq += 1
                self._frame = FrameCache(
                    seq=int(self._seq),
                    frame_header=header,
                    rgb_image=np.asarray(rgb, dtype=np.uint8),
                    depth_image_m=None if depth is None else np.asarray(depth, dtype=np.float32),
                    viewer_overlay=dict(overlay),
                    last_frame_monotonic=time.monotonic(),
                )
                self._decode_ok += 1
                self._update_stream_stalled(self._frame)

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

    def _reset_shm(self) -> None:
        if self._owns_shm and self._shm_ring is not None:
            with suppress(Exception):
                self._shm_ring.close()
        self._shm_ring = None

    def _decode_rgb(self, metadata: dict[str, object]) -> np.ndarray | None:
        if self._shm_ring is not None and isinstance(metadata.get("rgb_ref"), dict):
            return decode_ndarray(self._shm_ring.read(ref_from_dict(metadata["rgb_ref"])))
        return None

    def _decode_depth(self, metadata: dict[str, object]) -> np.ndarray | None:
        if self._shm_ring is not None and isinstance(metadata.get("depth_ref"), dict):
            return decode_ndarray(self._shm_ring.read(ref_from_dict(metadata["depth_ref"])))
        return None

    def _stream_fields(self, frame: FrameCache) -> dict[str, object]:
        age_ms = frame_age_ms(frame)
        stream_stalled = self._update_stream_stalled(frame)
        return {
            "stream_stalled": bool(stream_stalled),
            "streamStalled": bool(stream_stalled),
            "last_good_frame_age_ms": None if age_ms is None else round(float(age_ms), 3),
            "lastGoodFrameAgeMs": None if age_ms is None else round(float(age_ms), 3),
        }

    def _update_stream_stalled(self, frame: FrameCache | None) -> bool:
        stalled = is_frame_stale(frame, stale_after_sec=self.config.stale_frame_timeout_sec)
        if stalled != self._stream_stalled:
            self._stream_stalled = bool(stalled)
            self._stale_transitions += 1
        return self._stream_stalled

    def has_fresh_frame(self) -> bool:
        frame = self._frame
        if frame is None:
            return False
        return not self._update_stream_stalled(frame)

    def has_frame(self) -> bool:
        return self._frame is not None

    def shm_overwrite_drops(self) -> int:
        return int(self._shm_overwrite_drops)
