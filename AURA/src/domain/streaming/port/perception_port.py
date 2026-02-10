from typing import Protocol, Iterable, Any

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