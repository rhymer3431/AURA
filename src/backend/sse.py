"""Simple SSE broadcaster for dashboard state updates."""

from __future__ import annotations

import asyncio
import json

from aiohttp import web


class SseBroadcaster:
    def __init__(self):
        self._queues: set[asyncio.Queue[str]] = set()

    async def subscribe(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._queues.add(queue)
        try:
            while True:
                payload = await queue.get()
                await response.write(payload.encode("utf-8"))
        except asyncio.CancelledError:
            raise
        except ConnectionResetError:
            pass
        finally:
            self._queues.discard(queue)
        return response

    async def publish_state(self, state: dict[str, object]) -> None:
        payload = f"event: state\ndata: {json.dumps(state, ensure_ascii=False)}\n\n"
        for queue in list(self._queues):
            await queue.put(payload)
