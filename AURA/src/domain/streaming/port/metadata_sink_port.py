from typing import Protocol, Any, Dict

class MetadataSinkPort(Protocol):
    """Outbound metadata sink (WebSocket/WebRTC/etc)."""

    async def send_metadata(self, metadata: Dict[str, Any]) -> None:
        ...
