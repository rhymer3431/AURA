from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Dict

from .contracts import Plan, plan_from_json

try:
    from services.planner_server.schema import generate_stub_plan, validate_plan_payload
except Exception:  # pragma: no cover - optional import path
    generate_stub_plan = None
    validate_plan_payload = None


class PlannerClient:
    def __init__(self, cfg: Dict) -> None:
        self.url = str(cfg.get("url", "http://127.0.0.1:8088/plan"))
        self.timeout_s = float(cfg.get("timeout_s", 5.0))
        self.min_interval_s = float(cfg.get("min_interval_s", 1.0))
        self.mock_fallback = bool(cfg.get("mock_fallback", True))
        self._last_call_ts = 0.0
        self._degrade_level = 0

    def set_degrade_level(self, level: int) -> None:
        self._degrade_level = max(0, int(level))
        logging.info("Planner degrade level set to %s", self._degrade_level)

    async def create_plan(self, user_command: str, world_state: Dict) -> Plan:
        await self._respect_rate_limit()
        payload = {"user_command": user_command, "world_state": world_state}

        try:
            raw = await asyncio.to_thread(self._call_server, payload)
            self._last_call_ts = time.time()
            if validate_plan_payload is not None:
                validate_plan_payload(raw)
            return plan_from_json(raw)
        except Exception as exc:
            logging.warning("Planner server call failed. Falling back to stub: %s", exc)
            if not self.mock_fallback:
                raise
            return self._fallback_plan(user_command, world_state)

    async def _respect_rate_limit(self) -> None:
        scale = 1.0
        if self._degrade_level >= 1:
            scale = 2.0
        if self._degrade_level >= 2:
            scale = 4.0
        if self._degrade_level >= 3:
            scale = 6.0
        wait_s = (self.min_interval_s * scale) - (time.time() - self._last_call_ts)
        if wait_s > 0:
            await asyncio.sleep(wait_s)

    def _call_server(self, payload: Dict) -> Dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Plan-Mode": "cpu_throttle" if self._degrade_level >= 1 else "default",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                content = response.read().decode("utf-8")
            return json.loads(content)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Planner server HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Planner server unreachable: {exc}") from exc

    def _fallback_plan(self, user_command: str, world_state: Dict) -> Plan:
        if generate_stub_plan is not None:
            payload = generate_stub_plan(user_command, world_state)
            return plan_from_json(payload)

        object_name = "apple"
        payload = {
            "plan": [
                {"skill": "locate", "args": {"object": object_name}},
                {"skill": "navigate", "args": {"target": "object", "object": object_name}},
                {"skill": "pick", "args": {"object": object_name}},
                {"skill": "return", "args": {"target": "start"}},
                {"skill": "inspect", "args": {"object": object_name}},
            ],
            "notes": "local fallback plan",
        }
        return plan_from_json(payload)

