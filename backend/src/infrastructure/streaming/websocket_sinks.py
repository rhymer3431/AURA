import cv2
from fastapi import WebSocket
from typing import Dict

from domain.streaming.ports import VideoSinkPort, MetadataSinkPort


class WebSocketVideoSink(VideoSinkPort):
    def __init__(self, websocket: WebSocket, frame_max_width: int, jpeg_quality: int):
        self.websocket = websocket
        self.frame_max_width = frame_max_width
        self.jpeg_quality = jpeg_quality

    async def send_frame(self, frame_idx: int, frame_bgr) -> None:
        send_frame = frame_bgr
        h, w, _ = send_frame.shape
        if self.frame_max_width > 0 and w > self.frame_max_width:
            scale = self.frame_max_width / float(w)
            new_size = (self.frame_max_width, int(h * scale))
            send_frame = cv2.resize(send_frame, new_size, interpolation=cv2.INTER_AREA)

        encoded, buffer = cv2.imencode(
            ".jpg",
            send_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(self.jpeg_quality, 95))],
        )
        if encoded:
            await self.websocket.send_bytes(buffer.tobytes())


class WebSocketMetadataSink(MetadataSinkPort):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send_metadata(self, metadata: Dict):
        await self.websocket.send_json(metadata)
