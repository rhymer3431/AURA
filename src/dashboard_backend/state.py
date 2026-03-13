from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
import time
from typing import Any

from aiohttp import ClientSession

from webrtc.subscriber import ObservationSubscriber

from .config import DashboardBackendConfig
from .log_tailer import LogTailer
from .process_manager import ProcessManager
from .runtime_control import RuntimeControlClient


class StateAggregator:
    def __init__(
        self,
        config: DashboardBackendConfig,
        *,
        process_manager: ProcessManager,
        subscriber: ObservationSubscriber,
        control_client: RuntimeControlClient,
        session_manager,
        log_tailer: LogTailer,
    ) -> None:
        self.config = config
        self.process_manager = process_manager
        self.subscriber = subscriber
        self.control_client = control_client
        self.session_manager = session_manager
        self.log_tailer = log_tailer
        self._listener = subscriber.add_listener()
        self._tasks: list[asyncio.Task[None]] = []
        self._client: ClientSession | None = None
        self._runtime_snapshot: dict[str, object] = {}
        self._detector_capability: dict[str, object] = {}
        self._last_status: dict[str, object] = {}
        self._service_state: dict[str, dict[str, object]] = {}
        self._event_logs: deque[dict[str, object]] = deque(maxlen=200)
        self._state_listeners: list[asyncio.Queue[dict[str, object]]] = []
        self._state: dict[str, object] = {}

    async def start(self) -> None:
        if self._client is not None:
            return
        self._client = ClientSession()
        try:
            self._tasks = [
                asyncio.create_task(self._event_loop(), name="dashboard-state-events"),
                asyncio.create_task(self._poll_loop(), name="dashboard-state-poll"),
            ]
            await self.force_refresh()
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        self.subscriber.remove_listener(self._listener)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        if self._client is not None:
            await self._client.close()
            self._client = None

    def snapshot(self) -> dict[str, object]:
        return dict(self._state)

    def add_listener(self) -> asyncio.Queue[dict[str, object]]:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=8)
        self._state_listeners.append(queue)
        return queue

    def remove_listener(self, queue: asyncio.Queue[dict[str, object]]) -> None:
        self._state_listeners = [item for item in self._state_listeners if item is not queue]

    def get_recent_logs(self, *, limit: int = 200) -> list[dict[str, object]]:
        file_logs = self.log_tailer.get_recent(limit=max(limit, 1))
        event_logs = list(self._event_logs)[-max(limit, 1) :]
        return [*file_logs, *event_logs][-max(limit, 1) :]

    async def force_refresh(self) -> None:
        await self._refresh_external_services()
        self._state = self._build_state()
        self._broadcast_state()

    async def _event_loop(self) -> None:
        while True:
            event = await self._listener.get()
            self._consume_gateway_event(event.kind, event.payload)
            await self.force_refresh()

    async def _poll_loop(self) -> None:
        interval = max(float(self.config.health_poll_interval_sec), 0.2)
        while True:
            await asyncio.sleep(interval)
            await self.force_refresh()

    def _consume_gateway_event(self, kind: str, payload: dict[str, object]) -> None:
        if kind == "health" and str(payload.get("component", "")) == "aura_runtime":
            details = payload.get("details")
            if isinstance(details, dict) and isinstance(details.get("snapshot"), dict):
                self._runtime_snapshot = dict(details["snapshot"])
        elif kind == "capability" and str(payload.get("component", "")) == "detector":
            self._detector_capability = dict(payload)
        elif kind == "status":
            self._last_status = dict(payload)
        elif kind == "notice":
            self._event_logs.append(
                {
                    "source": str(payload.get("component", "runtime")),
                    "stream": "event",
                    "level": str(payload.get("level", "info")),
                    "message": str(payload.get("notice", "")),
                    "details": dict(payload.get("details", {})) if isinstance(payload.get("details"), dict) else {},
                    "timestampNs": int(payload.get("timestamp_ns", 0) or 0),
                }
            )

    async def _refresh_external_services(self) -> None:
        request = self.process_manager.current_request
        required = set() if request is None else request.required_process_names()
        navdp_health_url, navdp_debug_url = self.process_manager.service_urls("navdp")
        self._service_state["navdp"] = await self._service_snapshot(
            name="navdp",
            health_url=navdp_health_url,
            debug_url=navdp_debug_url,
            required="navdp" in required,
        )
        dual_health_url, dual_debug_url = self.process_manager.service_urls("dual")
        self._service_state["dual"] = await self._service_snapshot(
            name="dual",
            health_url=dual_health_url,
            debug_url=dual_debug_url,
            required="dual" in required,
        )

    async def _service_snapshot(
        self,
        *,
        name: str,
        health_url: str,
        debug_url: str,
        required: bool,
    ) -> dict[str, object]:
        process = next((item for item in self.process_manager.snapshot() if item["name"] == name), None)
        if process is None:
            return {"name": name, "status": "unknown"}
        if process["state"] == "not_required":
            return {"name": name, "status": "not_required", "healthUrl": health_url, "debug": {}}
        if process["state"] != "running":
            return {"name": name, "status": "down" if required else "inactive", "healthUrl": health_url, "debug": {}}
        started_at = time.perf_counter()
        health = await self._http_json(health_url)
        latency_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
        debug = await self._http_json(debug_url)
        status = "ok" if health is not None else ("down" if required else "inactive")
        return {
            "name": name,
            "status": status,
            "healthUrl": health_url,
            "latencyMs": latency_ms,
            "health": health or {},
            "debug": debug or {},
        }

    async def _http_json(self, url: str) -> dict[str, object] | None:
        if self._client is None:
            return None
        try:
            async with self._client.get(url, timeout=2.0) as response:
                if response.status != 200:
                    return None
                payload = await response.json()
                return dict(payload) if isinstance(payload, dict) else None
        except Exception:
            return None

    def _build_state(self) -> dict[str, object]:
        request = self.process_manager.current_request
        session_payload = None if request is None else request.to_public_dict()
        processes = self.process_manager.snapshot()
        runtime = dict(self._runtime_snapshot.get("planner", {})) if isinstance(self._runtime_snapshot.get("planner"), dict) else {}
        if isinstance(self._runtime_snapshot.get("modes"), dict):
            runtime["modes"] = dict(self._runtime_snapshot["modes"])
        if self._last_status:
            runtime["lastStatusEvent"] = dict(self._last_status)
        sensors = dict(self._runtime_snapshot.get("sensor", {})) if isinstance(self._runtime_snapshot.get("sensor"), dict) else {}
        perception = dict(self._runtime_snapshot.get("perception", {})) if isinstance(self._runtime_snapshot.get("perception"), dict) else {}
        if self._detector_capability:
            perception["detectorCapability"] = dict(self._detector_capability)
        memory = dict(self._runtime_snapshot.get("memory", {})) if isinstance(self._runtime_snapshot.get("memory"), dict) else {}
        transport = dict(self._runtime_snapshot.get("transport", {})) if isinstance(self._runtime_snapshot.get("transport"), dict) else {}
        frame = self.subscriber.current_frame
        transport.update(
            {
                "viewerEnabled": bool(request.viewer_enabled) if request is not None else False,
                "frameAgeMs": self.subscriber.last_frame_age_ms(),
                "frameSeq": None if frame is None else int(frame.seq),
                "frameAvailable": frame is not None,
                "peerActive": self.session_manager.active_session is not None,
                "peerSessionId": None if self.session_manager.active_session is None else self.session_manager.active_session.session_id,
                "peerTrackRoles": []
                if self.session_manager.active_session is None
                else list(self.session_manager.active_session.track_roles),
                "busHealth": self.control_client.transport_health_snapshot(),
            }
        )
        services = {
            "navdp": dict(self._service_state.get("navdp", {})),
            "dual": dict(self._service_state.get("dual", {})),
            "system2": next((item for item in processes if item["name"] == "system2"), {"name": "system2", "state": "inactive"}),
        }
        return {
            "timestamp": time.time(),
            "session": {
                "active": request is not None,
                "startedAt": self.process_manager.session_started_at,
                "config": session_payload,
                "lastEvent": None if not self._event_logs else dict(self._event_logs[-1]),
            },
            "processes": processes,
            "runtime": runtime,
            "sensors": sensors,
            "perception": perception,
            "memory": memory,
            "services": services,
            "transport": transport,
            "logs": self.get_recent_logs(limit=200),
        }

    def _broadcast_state(self) -> None:
        alive: list[asyncio.Queue[dict[str, object]]] = []
        for queue in self._state_listeners:
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            try:
                queue.put_nowait(dict(self._state))
            except asyncio.QueueFull:
                continue
            alive.append(queue)
        self._state_listeners = alive
