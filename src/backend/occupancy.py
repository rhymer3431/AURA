"""Occupancy metadata provider for the backend."""

from __future__ import annotations

from aiohttp import web


def build_occupancy_payload(scene_preset: str) -> dict[str, object]:
    return {
        "available": False,
        "scenePreset": str(scene_preset),
        "reason": "No occupancy map asset is configured for this scene preset.",
    }


async def handle_image(_request: web.Request) -> web.Response:
    return web.Response(status=404, text="occupancy image unavailable")
