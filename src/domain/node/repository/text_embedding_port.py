from abc import ABC, abstractmethod
from typing import Any


class TextEmbeddingPort(ABC):
    """Abstraction for text embedding backends (e.g., CLIP, LLM)."""

    @abstractmethod
    def encode(self, text: str) -> Any:
        raise NotImplementedError
