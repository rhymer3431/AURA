import asyncio
import json
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from aiortc import RTCDataChannel, RTCPeerConnection, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from src.domain.streaming.port import VideoSinkPort, MetadataSinkPort


class PerceptionVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(
        self,
        frame_queue: "asyncio.Queue[Optional[Tuple[int, np.ndarray]]]",
        stop_event: asyncio.Event,
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_event = stop_event

    async def recv(self) -> VideoFrame:
        if self.stop_event.is_set():
            raise MediaStreamError("stream closed")

        item = await self.frame_queue.get()
        if item is None or self.stop_event.is_set():
            raise MediaStreamError("stream closed")

        _, frame_rgb = item
        frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        frame.pts, frame.time_base = await self.next_timestamp()
        return frame


class WebRtcVideoSink(VideoSinkPort):
    def __init__(
        self,
        frame_queue: "asyncio.Queue[Optional[Tuple[int, np.ndarray]]]",
        stop_event: asyncio.Event,
        frame_max_width: int,
    ):
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.frame_max_width = frame_max_width

    async def send_frame(self, frame_idx: int, frame_bgr) -> None:
        send_frame = frame_bgr
        h, w, _ = send_frame.shape
        if self.frame_max_width > 0 and w > self.frame_max_width:
            scale = self.frame_max_width / float(w)
            new_size = (self.frame_max_width, int(h * scale))
            send_frame = cv2.resize(send_frame, new_size, interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(send_frame, cv2.COLOR_BGR2RGB)

        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        await self.frame_queue.put((frame_idx, frame_rgb))


class WebRtcMetadataSink(MetadataSinkPort):
    def __init__(self, channel_getter):
        self.channel_getter = channel_getter

    async def send_metadata(self, metadata: Dict):
        channel: Optional[RTCDataChannel] = self.channel_getter()

        if not channel or channel.readyState != "open":
            return

        try:
            channel.send(json.dumps(metadata))
        except Exception:
            # Network is not ready; skip instead of crashing
            return
