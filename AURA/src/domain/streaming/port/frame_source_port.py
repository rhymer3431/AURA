from typing import Optional, Protocol, Any


class FrameSourcePort(Protocol):
    """Inbound frame source (video file, ROS2 topic, etc)."""

    is_live: bool

    async def read_frame(self) -> Optional[Any]:
        ...

    async def close(self) -> None:
        ...
