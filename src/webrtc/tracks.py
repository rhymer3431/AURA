from __future__ import annotations

import asyncio
from fractions import Fraction

import numpy as np

from common.depth_visualization import depth_to_heatmap_bgr

from .subscriber import ObservationSubscriber

try:
    from aiortc import VideoStreamTrack
    from av import VideoFrame
except Exception as exc:  # noqa: BLE001
    VideoStreamTrack = object  # type: ignore[assignment]
    VideoFrame = None  # type: ignore[assignment]
    _VIDEO_IMPORT_ERROR = exc
else:
    _VIDEO_IMPORT_ERROR = None


def _require_video_dependencies() -> None:
    if _VIDEO_IMPORT_ERROR is not None:
        raise RuntimeError("aiortc and av are required for WebRTC video tracks.") from _VIDEO_IMPORT_ERROR


class _LatestFrameTrack(VideoStreamTrack):  # type: ignore[misc]
    def __init__(self, subscriber: ObservationSubscriber, *, fps: float, track_role: str) -> None:
        _require_video_dependencies()
        super().__init__()
        self._subscriber = subscriber
        self._fps = max(float(fps), 1.0)
        self.track_role = str(track_role)
        self._frame_period = 1.0 / self._fps
        self._next_frame_at: float | None = None
        self._pts = 0

    async def recv(self):
        now = asyncio.get_running_loop().time()
        if self._next_frame_at is None:
            self._next_frame_at = now + self._frame_period
        else:
            wait_time = self._next_frame_at - now
            if wait_time > 0.0:
                await asyncio.sleep(wait_time)
            self._next_frame_at = max(self._next_frame_at + self._frame_period, asyncio.get_running_loop().time())
        image, pixel_format = self._render_image()
        frame = VideoFrame.from_ndarray(image, format=pixel_format)
        frame.pts = int(self._pts)
        frame.time_base = Fraction(1, 90000)
        self._pts += int(round(90000.0 / self._fps))
        return frame

    def _render_image(self) -> tuple[np.ndarray, str]:
        raise NotImplementedError

    def _frame_shape(self) -> tuple[int, int]:
        current = self._subscriber.current_frame
        if current is not None and current.rgb_image.ndim == 3:
            return int(current.rgb_image.shape[0]), int(current.rgb_image.shape[1])
        return 240, 320


class RgbVideoTrack(_LatestFrameTrack):
    def __init__(self, subscriber: ObservationSubscriber, *, fps: float) -> None:
        super().__init__(subscriber, fps=fps, track_role="rgb")

    def _render_image(self) -> tuple[np.ndarray, str]:
        current = self._subscriber.current_frame
        if current is not None:
            return np.asarray(current.rgb_image, dtype=np.uint8), "rgb24"
        height, width = self._frame_shape()
        return np.zeros((height, width, 3), dtype=np.uint8), "rgb24"


class DepthPreviewVideoTrack(_LatestFrameTrack):
    def __init__(self, subscriber: ObservationSubscriber, *, fps: float) -> None:
        super().__init__(subscriber, fps=fps, track_role="depth")

    def _render_image(self) -> tuple[np.ndarray, str]:
        current = self._subscriber.current_frame
        if current is not None and current.depth_image_m is not None:
            preview = depth_to_heatmap_bgr(current.depth_image_m, depth_max_m=5.0)
            return np.asarray(preview, dtype=np.uint8), "bgr24"
        height, width = self._frame_shape()
        return np.zeros((height, width, 3), dtype=np.uint8), "bgr24"
