from typing import Protocol

class VideoSinkPort(Protocol):
    """Outbound video sink (WebSocket/WebRTC/etc)."""

    async def send_frame(self, frame_idx: int, frame_bgr) -> None:
        ...