from __future__ import annotations

from collections import defaultdict, deque

from .base import BusRecord, MessageBus
from .codec import decode_envelope, encode_envelope
from .messages import MessagePayload


class ZmqBus(MessageBus):
    def __init__(self, endpoint: str, *, bind: bool) -> None:
        try:
            import zmq
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("pyzmq is required for ZmqBus.") from exc

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PAIR)
        if bind:
            self._socket.bind(str(endpoint))
        else:
            self._socket.connect(str(endpoint))
        self._buffer: dict[str, deque[BusRecord]] = defaultdict(deque)

    def publish(self, topic: str, message: MessagePayload) -> None:
        self._socket.send_multipart(list(encode_envelope(topic, message)))

    def poll(self, topic: str, *, max_items: int | None = None) -> list[BusRecord]:
        requested_topic = str(topic)
        while True:
            try:
                raw_topic, raw_payload = self._socket.recv_multipart(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            parsed_topic, parsed_message = decode_envelope(raw_topic, raw_payload)
            self._buffer[parsed_topic].append(BusRecord(topic=parsed_topic, message=parsed_message))

        queue = self._buffer[requested_topic]
        if max_items is None or max_items < 0:
            limit = len(queue)
        else:
            limit = min(len(queue), int(max_items))
        return [queue.popleft() for _ in range(limit)]

    def close(self) -> None:
        self._socket.close(linger=0)
