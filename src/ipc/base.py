from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .messages import MessagePayload


@dataclass(frozen=True)
class BusRecord:
    topic: str
    message: MessagePayload


class MessageBus(ABC):
    @abstractmethod
    def publish(self, topic: str, message: MessagePayload) -> None:
        raise NotImplementedError

    @abstractmethod
    def poll(self, topic: str, *, max_items: int | None = None) -> list[BusRecord]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
