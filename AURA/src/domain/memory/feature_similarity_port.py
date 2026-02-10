# src/domain/memory/similarity_port.py
from typing import Any, Protocol


class FeatureSimilarityPort(Protocol):
    def get_embedding(self, image: Any) -> Any:
        """Returns L2 normalized embedding tensor."""
        ...

    def cosine_similarity(self, a: Any, b: Any) -> float:
        """Cosine similarity score."""
        ...