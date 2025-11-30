from typing import Optional

from domain.memory.entity.memory_state import MemoryItem, MemoryState
from domain.memory.repository.memory_storage_port import MemoryStoragePort


class MemoryService:
    def __init__(self, storage: MemoryStoragePort):
        self.storage = storage

    def save(self, item: MemoryItem) -> None:
        self.storage.save(item)

    def load(self, key: str) -> Optional[MemoryItem]:
        return self.storage.load(key)

    def snapshot(self) -> MemoryState:
        return self.storage.snapshot()
