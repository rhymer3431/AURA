"""Read inference-system state for the backend."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fetch_inference_state(base_url: str, *, timeout_s: float = 5.0) -> dict[str, Any]:
    url = f"{str(base_url).rstrip('/')}/models/state"
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"ok": True, "state": payload}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "error": f"http_{exc.code}: {detail}"}
    except URLError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def fetch_stack_state(base_url: str, *, timeout_s: float = 2.0) -> dict[str, Any]:
    return fetch_inference_state(base_url, timeout_s=timeout_s)
