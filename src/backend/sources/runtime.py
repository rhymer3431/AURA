"""Runtime service client helpers for the backend."""

from __future__ import annotations

from typing import Any

from aiohttp import ClientError, ClientSession


async def fetch_runtime_state(http: ClientSession | None, base_url: str, *, timeout_s: float = 2.0) -> dict[str, Any]:
    if http is None:
        return {"ok": False, "error": "dashboard_http_client_unavailable"}
    url = f"{str(base_url).rstrip('/')}/session/state"
    try:
        async with http.get(url, timeout=timeout_s) as response:
            payload = await response.json()
        if not isinstance(payload, dict):
            return {"ok": False, "error": "runtime returned a non-object payload"}
        return {"ok": response.ok, "state": payload}
    except ClientError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def post_runtime_session(
    http: ClientSession | None,
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    if http is None:
        return {"ok": False, "error": "dashboard_http_client_unavailable"}
    url = f"{str(base_url).rstrip('/')}{path}"
    try:
        async with http.post(url, json=payload, timeout=timeout_s) as response:
            response_payload = await response.json()
        if not isinstance(response_payload, dict):
            return {"ok": False, "error": "runtime returned a non-object payload"}
        response_payload.setdefault("ok", response.ok)
        return response_payload
    except ClientError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
