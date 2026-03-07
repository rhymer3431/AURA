from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock

from .base import BusRecord, MessageBus
from .messages import MessagePayload


class InprocBus(MessageBus):
    def __init__(self) -> None:
        self._queues: dict[str, deque[BusRecord]] = defaultdict(deque)
        self._lock = Lock()

    def publish(self, topic: str, message: MessagePayload) -> None:
        with self._lock:
            self._queues[str(topic)].append(BusRecord(topic=str(topic), message=message))

    def poll(self, topic: str, *, max_items: int | None = None) -> list[BusRecord]:
        with self._lock:
            queue = self._queues[str(topic)]
            if max_items is None or max_items < 0:
                limit = len(queue)
            else:
                limit = min(len(queue), int(max_items))
            return [queue.popleft() for _ in range(limit)]

    def close(self) -> None:
        with self._lock:
            self._queues.clear()
