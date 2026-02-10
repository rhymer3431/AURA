from typing import Protocol, Iterable, Tuple, Any, Dict

class ScenePlanPort(Protocol):
    """Interface for scheduling and retrieving scene plans (LLM results)."""

    def submit(self, frame_idx: int, simple_scene_graph) -> None:
        ...

    def poll_results(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        ...
