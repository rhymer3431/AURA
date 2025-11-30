from abc import ABC, abstractmethod
from typing import Optional

from domain.memory.entity.memory_state import MemoryItem, MemoryState


class MemoryStoragePort(ABC):
    @abstractmethod
    def save(self, item: MemoryItem) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, key: str) -> Optional[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> MemoryState:
        raise NotImplementedError
