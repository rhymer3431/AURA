from typing import Protocol, Iterable, Tuple, Any, Dict


class PerceptionPort(Protocol):
    """Perception facade expected by the application layer."""

    def process_frame(self, frame_bgr, frame_idx: int, run_grin: bool, max_entities: int):
        ...

    def build_simple_scene_graph_frame(self, sg_frame):
        ...

    def update_focus_classes(self, focus_targets: Iterable[str]) -> None:
        ...

    @property
    def ltm_entities(self) -> Any:
        ...


class ScenePlanPort(Protocol):
    """Interface for scheduling and retrieving scene plans (LLM results)."""

    def submit(self, frame_idx: int, simple_scene_graph) -> None:
        ...

    def poll_results(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        ...


class VideoSinkPort(Protocol):
    """Outbound video sink (WebSocket/WebRTC/etc)."""

    async def send_frame(self, frame_idx: int, frame_bgr) -> None:
        ...


class MetadataSinkPort(Protocol):
    """Outbound metadata sink (WebSocket/WebRTC/etc)."""

    async def send_metadata(self, metadata: Dict[str, Any]) -> None:
        ...
