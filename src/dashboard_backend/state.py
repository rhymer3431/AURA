from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
import time
from typing import Any

from aiohttp import ClientSession

from schemas.world_state import WorldStateSnapshot
from server.snapshot_adapter import SnapshotAdapter
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
        self._world_state: WorldStateSnapshot | None = None
        self._detector_capability: dict[str, object] = {}
        self._last_status: dict[str, object] = {}
        self._service_state: dict[str, dict[str, object]] = {}
        self._event_logs: deque[dict[str, object]] = deque(maxlen=200)
        self._state_listeners: list[asyncio.Queue[dict[str, object]]] = []
        self._state: dict[str, object] = {}

    @staticmethod
    def _architecture_node(
        *,
        name: str,
        status: str,
        summary: str,
        detail: str = "",
        required: bool = True,
        latency_ms: float | None = None,
        metrics: dict[str, object] | None = None,
        core: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": name,
            "status": str(status),
            "summary": str(summary),
            "detail": str(detail),
            "required": bool(required),
            "metrics": {} if metrics is None else dict(metrics),
        }
        if latency_ms is not None:
            payload["latencyMs"] = float(latency_ms)
        if core is not None:
            payload["core"] = dict(core)
        return payload

    @staticmethod
    def _node_detail(*parts: object) -> str:
        chunks = [str(part).strip() for part in parts if str(part).strip() != ""]
        return " · ".join(chunks)

    @staticmethod
    def _process_status(process: dict[str, object] | None) -> str:
        state = "" if process is None else str(process.get("state", "unknown"))
        if state == "running":
            return "ok"
        if state == "exited":
            return "failed"
        if state == "not_required":
            return "not_required"
        if state in {"stopped", "inactive"}:
            return "inactive"
        return "unknown"

    @staticmethod
    def _system2_output_snapshot(dual_service: dict[str, object]) -> dict[str, object] | None:
        debug = dict(dual_service.get("debug", {})) if isinstance(dual_service.get("debug"), dict) else {}
        stats = dict(debug.get("stats", {})) if isinstance(debug.get("stats"), dict) else {}
        session = dict(debug.get("system2_session", {})) if isinstance(debug.get("system2_session"), dict) else {}

        raw_text = str(session.get("last_output", "")).strip() or str(stats.get("last_s2_raw_text", "")).strip()
        reason = str(session.get("last_reason", "")).strip() or str(stats.get("last_s2_reason", "")).strip()
        decision_mode = str(session.get("last_decision_mode", "")).strip() or str(stats.get("last_s2_mode", "")).strip()
        instruction = str(session.get("instruction", "")).strip() or str(debug.get("instruction", "")).strip()
        history_frame_ids = session.get("last_history_frame_ids", stats.get("last_s2_history_frame_ids", []))
        normalized_history = (
            [int(item) for item in history_frame_ids]
            if isinstance(history_frame_ids, list)
            else []
        )
        needs_requery = bool(session.get("last_needs_requery", stats.get("last_s2_needs_requery", False)))
        requested_stop = bool(stats.get("last_s2_requested_stop", False))
        effective_stop = bool(stats.get("last_s2_effective_stop", False))
        latency_ms_raw = stats.get("last_s2_latency_ms")
        latency_ms = float(latency_ms_raw) if isinstance(latency_ms_raw, (int, float)) else None
        has_output = (
            raw_text != ""
            or reason != ""
            or normalized_history != []
            or requested_stop
            or effective_stop
            or decision_mode not in {"", "wait"}
        )

        if not has_output:
            return None

        payload: dict[str, object] = {
            "rawText": raw_text,
            "reason": reason,
            "decisionMode": decision_mode,
            "needsRequery": needs_requery,
            "historyFrameIds": normalized_history,
            "requestedStop": requested_stop,
            "effectiveStop": effective_stop,
            "instruction": instruction,
        }
        if latency_ms is not None:
            payload["latencyMs"] = latency_ms
        return payload

    def _system2_service_state(
        self,
        *,
        session_active: bool,
        processes: list[dict[str, object]],
        dual_service: dict[str, object],
    ) -> dict[str, object]:
        system2_process = next((item for item in processes if item.get("name") == "system2"), None)
        process_status = self._process_status(system2_process)
        dual_status = str(dual_service.get("status", "inactive"))
        output = self._system2_output_snapshot(dual_service)

        if not session_active:
            status = "inactive"
        elif process_status == "failed" or dual_status in {"down", "failed"}:
            status = "down"
        elif output is None:
            status = "awaiting_first_decision"
        elif process_status == "ok" and dual_status == "ok":
            status = "ok"
        else:
            status = "degraded"

        payload: dict[str, object] = {
            "name": "system2",
            "status": status,
            "healthUrl": "" if system2_process is None else str(system2_process.get("healthUrl", "")),
            "output": output,
        }
        if output is not None and isinstance(output.get("latencyMs"), (int, float)):
            payload["latencyMs"] = float(output["latencyMs"])
        return payload

    def _build_architecture_state(
        self,
        *,
        session_active: bool,
        processes: list[dict[str, object]],
        services: dict[str, object],
        transport_state: dict[str, object],
    ) -> dict[str, object]:
        world_state = WorldStateSnapshot() if self._world_state is None else self._world_state
        process_map = {str(item.get("name", "")): dict(item) for item in processes if isinstance(item, dict)}
        nav_service = dict(services.get("navdp", {})) if isinstance(services.get("navdp"), dict) else {}
        dual_service = dict(services.get("dual", {})) if isinstance(services.get("dual"), dict) else {}
        bus_health = dict(transport_state.get("busHealth", {})) if isinstance(transport_state.get("busHealth"), dict) else {}
        system2_output = self._system2_output_snapshot(dual_service)

        planner_mode = str(world_state.mode or world_state.task.mode or "")
        task_state = str(world_state.task.state or ("active" if session_active else "idle"))
        task_id = str(world_state.task.task_id)
        active_command_type = str(world_state.execution.active_command_type)
        recovery_state = str(world_state.safety.recovery_state.current_state)
        recovery_reason = str(world_state.safety.recovery_state.last_trigger_reason)
        planner_control_mode = str(world_state.planning.planner_control_mode)
        frame_available = bool(transport_state.get("frameAvailable", False))
        frame_source = str(world_state.robot.source)
        frame_id = int(world_state.robot.frame_id)
        frame_age_ms = transport_state.get("frameAgeMs")
        nav_latency = nav_service.get("latencyMs")
        s2_latency = dual_service.get("latencyMs")
        detection_enabled = bool(world_state.runtime.detection_enabled)
        memory_store = bool(world_state.runtime.memory_store)
        viewer_publish = bool(world_state.runtime.viewer_publish)
        peer_active = bool(transport_state.get("peerActive", False))
        goal_distance = world_state.execution.locomotion_proposal_summary.get("goal_distance_m")
        yaw_error = world_state.execution.locomotion_proposal_summary.get("yaw_error_rad")
        system2_required = session_active

        if not session_active and not frame_available:
            gateway_status = "inactive"
            gateway_summary = "Session idle"
        elif world_state.safety.sensor_unavailable or world_state.safety.timeout:
            gateway_status = "warning"
            gateway_summary = "Ingress degraded"
        elif frame_available or frame_id >= 0:
            gateway_status = "ok"
            gateway_summary = "Frame ingress active"
        else:
            gateway_status = "warning"
            gateway_summary = "Waiting for frame"
        gateway_detail = self._node_detail(
            f"frame {frame_id}" if frame_id >= 0 else "",
            f"{round(float(frame_age_ms))}ms" if isinstance(frame_age_ms, (int, float)) else "",
            frame_source,
        )

        if recovery_state == "FAILED":
            control_status = "failed"
        elif recovery_state in {"SAFE_STOP", "WAIT_SENSOR", "REPLAN_PENDING", "RECOVERY_TURN"}:
            control_status = "warning"
        elif session_active or task_state not in {"", "idle"}:
            control_status = "ok"
        else:
            control_status = "inactive"
        if not session_active and task_state == "idle":
            control_summary = "Ready"
        else:
            control_summary = self._node_detail(planner_mode or "runtime", task_state or "idle")
        control_detail = self._node_detail(
            f"recovery {recovery_state.lower()}",
            f"cmd {active_command_type or 'idle'}",
        )

        world_state_store_node = self._architecture_node(
            name="World State Store",
            status="ok" if session_active or frame_id >= 0 or task_id != "" else "inactive",
            summary=self._node_detail("task", task_state or "idle"),
            detail=self._node_detail("recovery", recovery_state.lower()),
            metrics={
                "taskId": task_id,
                "taskState": task_state,
                "mode": planner_mode,
            },
        )
        decision_engine_node = self._architecture_node(
            name="Decision Engine",
            status="inactive" if not session_active else ("warning" if recovery_state != "NORMAL" else "ok"),
            summary="Policy gating",
            detail=self._node_detail(recovery_reason or "stable"),
            metrics={
                "recoveryState": recovery_state,
                "recoveryReason": recovery_reason,
            },
        )
        planner_coordinator_node = self._architecture_node(
            name="Planner Coordinator",
            status=(
                "inactive"
                if not session_active
                else (
                    "ok"
                    if int(world_state.planning.plan_version) >= 0
                    or int(world_state.planning.traj_version) >= 0
                    or planner_control_mode != ""
                    else "warning"
                )
            ),
            summary=self._node_detail(
                f"plan v{int(world_state.planning.plan_version)}",
                f"traj v{int(world_state.planning.traj_version)}",
            ),
            detail=self._node_detail(f"control {planner_control_mode or 'idle'}"),
            metrics={
                "planVersion": int(world_state.planning.plan_version),
                "goalVersion": int(world_state.planning.goal_version),
                "trajVersion": int(world_state.planning.traj_version),
                "plannerControlMode": planner_control_mode,
            },
        )
        command_resolver_node = self._architecture_node(
            name="Command Resolver",
            status=(
                "failed"
                if recovery_state == "FAILED"
                else (
                    "inactive"
                    if not session_active and active_command_type == ""
                    else ("warning" if recovery_state == "SAFE_STOP" else ("ok" if active_command_type != "" else "warning"))
                )
            ),
            summary=self._node_detail("cmd", active_command_type or "idle"),
            detail=self._node_detail(
                str(world_state.execution.last_action_status.get("reason", "")) or "no active override",
            ),
            metrics={
                "activeCommandType": active_command_type,
                "lastStatus": str(world_state.execution.last_action_status.get("state", "")),
                "statusReason": str(world_state.execution.last_action_status.get("reason", "")),
            },
        )
        safety_supervisor_node = self._architecture_node(
            name="Safety Supervisor",
            status=(
                "failed"
                if recovery_state == "FAILED"
                else (
                    "warning"
                    if world_state.safety.safe_stop
                    or world_state.safety.stale
                    or world_state.safety.timeout
                    or world_state.safety.sensor_unavailable
                    else ("ok" if session_active else "inactive")
                )
            ),
            summary=self._node_detail("recovery", recovery_state.lower()),
            detail=self._node_detail(recovery_reason or "normal"),
            metrics={
                "safeStop": bool(world_state.safety.safe_stop),
                "stale": bool(world_state.safety.stale),
                "timeout": bool(world_state.safety.timeout),
                "sensorUnavailable": bool(world_state.safety.sensor_unavailable),
            },
        )

        if not detection_enabled:
            perception_status = "not_required"
            perception_summary = "Detection disabled"
        elif not session_active and not frame_available:
            perception_status = "inactive"
            perception_summary = "Waiting for frames"
        elif world_state.perception.detector_ready or str(world_state.perception.detector_backend) != "":
            perception_status = "ok"
            perception_summary = f"{int(world_state.perception.detection_count)} detections"
        else:
            perception_status = "warning"
            perception_summary = "Detector warming up"

        if not memory_store and not world_state.memory.memory_aware_task_active:
            memory_status = "not_required"
            memory_summary = "Memory store disabled"
        elif not session_active and int(world_state.memory.object_count) == 0 and int(world_state.memory.place_count) == 0:
            memory_status = "inactive"
            memory_summary = "No active memory task"
        else:
            memory_status = "ok"
            memory_summary = self._node_detail(
                f"{int(world_state.memory.object_count)} objects",
                f"{int(world_state.memory.place_count)} places",
            )

        system2_process = process_map.get("system2")
        dual_status = str(dual_service.get("status", "unknown"))
        if not system2_required:
            s2_status = "not_required"
            s2_summary = "Session inactive"
        elif system2_output is not None and self._process_status(system2_process) == "ok" and dual_status == "ok":
            s2_status = "ok"
            decision_mode = str(system2_output.get("decisionMode", "")).replace("_", " ").strip()
            s2_summary = f"Decision {decision_mode or 'ready'}"
        elif self._process_status(system2_process) == "ok" and dual_status == "ok":
            s2_status = "ok"
            s2_summary = "Standing by" if planner_mode != "NAV" else "NAV planning active"
        elif self._process_status(system2_process) in {"failed", "down"} or dual_status in {"down", "failed"}:
            s2_status = "failed"
            s2_summary = "S2 path unavailable"
        else:
            s2_status = "warning"
            s2_summary = "S2 path warming up"

        nav_required = session_active
        if not nav_required:
            nav_status = "inactive"
            nav_summary = "No active session"
        elif str(nav_service.get("status", "unknown")) == "ok":
            nav_status = "ok"
            nav_summary = f"traj v{int(world_state.planning.traj_version)}"
        elif str(nav_service.get("status", "")) in {"not_required", "inactive"}:
            nav_status = "warning"
            nav_summary = "Planner transport idle"
        else:
            nav_status = "failed"
            nav_summary = "Navigation service unavailable"

        if not session_active:
            locomotion_status = "inactive"
            locomotion_summary = "No active command"
        elif recovery_state == "FAILED":
            locomotion_status = "failed"
            locomotion_summary = "Execution failed"
        elif recovery_state in {"SAFE_STOP", "WAIT_SENSOR"}:
            locomotion_status = "warning"
            locomotion_summary = "Safety override active"
        elif active_command_type != "" or world_state.execution.locomotion_proposal_summary != {}:
            locomotion_status = "ok"
            locomotion_summary = active_command_type or "Proposal active"
        else:
            locomotion_status = "warning"
            locomotion_summary = "Awaiting proposal"

        if not session_active and not viewer_publish and not peer_active:
            telemetry_status = "inactive"
            telemetry_summary = "No active viewers"
        elif bus_health != {} or viewer_publish or peer_active:
            telemetry_status = "ok"
            telemetry_summary = "State mirror active"
        else:
            telemetry_status = "warning"
            telemetry_summary = "Telemetry degraded"

        return {
            "gateway": self._architecture_node(
                name="Robot Gateway",
                status=gateway_status,
                summary=gateway_summary,
                detail=gateway_detail,
                latency_ms=float(frame_age_ms) if isinstance(frame_age_ms, (int, float)) else None,
                metrics={
                    "frameId": frame_id,
                    "frameSource": frame_source,
                    "frameAgeMs": frame_age_ms,
                    "viewerEnabled": bool(transport_state.get("viewerEnabled", False)),
                    "peerActive": peer_active,
                },
            ),
            "mainControlServer": self._architecture_node(
                name="Main Control Server",
                status=control_status,
                summary=control_summary,
                detail=control_detail,
                metrics={
                    "mode": planner_mode,
                    "taskId": task_id,
                    "taskState": task_state,
                    "instruction": str(world_state.task.instruction),
                    "plannerControlMode": planner_control_mode,
                    "activeCommandType": active_command_type,
                    "recoveryState": recovery_state,
                    "recoveryReason": recovery_reason,
                },
                core={
                    "worldStateStore": world_state_store_node,
                    "decisionEngine": decision_engine_node,
                    "plannerCoordinator": planner_coordinator_node,
                    "commandResolver": command_resolver_node,
                    "safetySupervisor": safety_supervisor_node,
                },
            ),
            "modules": {
                "perception": self._architecture_node(
                    name="Perception",
                    status=perception_status,
                    summary=perception_summary,
                    detail=self._node_detail(
                        str(world_state.perception.detector_backend),
                        str(world_state.perception.detector_selected_reason),
                    ),
                    required=detection_enabled,
                    metrics={
                        "detectionCount": int(world_state.perception.detection_count),
                        "trackedDetectionCount": int(world_state.perception.tracked_detection_count),
                        "detectorBackend": str(world_state.perception.detector_backend),
                    },
                ),
                "memory": self._architecture_node(
                    name="Memory",
                    status=memory_status,
                    summary=memory_summary,
                    detail=self._node_detail(str(world_state.memory.scratchpad.get("taskState", "")) or "scratchpad idle"),
                    required=memory_store or bool(world_state.memory.memory_aware_task_active),
                    metrics={
                        "objectCount": int(world_state.memory.object_count),
                        "placeCount": int(world_state.memory.place_count),
                        "scratchpadState": str(world_state.memory.scratchpad.get("taskState", "")),
                    },
                ),
                "s2": self._architecture_node(
                    name="S2",
                    status=s2_status,
                    summary=s2_summary,
                    detail=self._node_detail(
                        str(system2_output.get("instruction", "")) if system2_output is not None else str(world_state.planning.active_instruction or "idle"),
                        str(system2_output.get("rawText", "")) if system2_output is not None else "",
                        str(system2_output.get("reason", "")) if system2_output is not None else "",
                        "pixel goal ready" if world_state.planning.system2_pixel_goal is not None else "",
                    ),
                    required=system2_required,
                    latency_ms=float(s2_latency) if isinstance(s2_latency, (int, float)) else None,
                    metrics={
                        "executionMode": str(world_state.mode),
                        "activeInstruction": str(world_state.planning.active_instruction),
                        "decisionMode": "" if system2_output is None else str(system2_output.get("decisionMode", "")),
                        "lastOutput": "" if system2_output is None else str(system2_output.get("rawText", "")),
                        "needsRequery": False if system2_output is None else bool(system2_output.get("needsRequery", False)),
                        "system2PixelGoal": None
                        if world_state.planning.system2_pixel_goal is None
                        else list(world_state.planning.system2_pixel_goal),
                    },
                ),
                "nav": self._architecture_node(
                    name="Nav",
                    status=nav_status,
                    summary=nav_summary,
                    detail=self._node_detail(
                        f"goal {goal_distance:.2f}m" if isinstance(goal_distance, (int, float)) else "",
                        str(nav_service.get("status", "")),
                    ),
                    required=nav_required,
                    latency_ms=float(nav_latency) if isinstance(nav_latency, (int, float)) else None,
                    metrics={
                        "planVersion": int(world_state.planning.plan_version),
                        "trajVersion": int(world_state.planning.traj_version),
                        "goalDistanceM": goal_distance,
                    },
                ),
                "locomotion": self._architecture_node(
                    name="Locomotion",
                    status=locomotion_status,
                    summary=locomotion_summary,
                    detail=self._node_detail(
                        f"yaw {yaw_error:.2f} rad" if isinstance(yaw_error, (int, float)) else "",
                        planner_control_mode,
                    ),
                    required=session_active,
                    metrics={
                        "activeCommandType": active_command_type,
                        "goalDistanceM": goal_distance,
                        "yawErrorRad": yaw_error,
                    },
                ),
                "telemetry": self._architecture_node(
                    name="Telemetry",
                    status=telemetry_status,
                    summary=telemetry_summary,
                    detail=self._node_detail(
                        "viewer publish" if viewer_publish else "",
                        "peer connected" if peer_active else "",
                    ),
                    required=session_active or viewer_publish,
                    latency_ms=float(frame_age_ms) if isinstance(frame_age_ms, (int, float)) else None,
                    metrics={
                        "controlEndpoint": str(bus_health.get("control_endpoint", "")),
                        "telemetryEndpoint": str(bus_health.get("telemetry_endpoint", "")),
                        "peerActive": peer_active,
                        "viewerPublish": viewer_publish,
                    },
                ),
            },
        }

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

    async def force_refresh(self, *, refresh_services: bool = True) -> None:
        if refresh_services:
            await self._refresh_external_services()
        self._state = self._build_state()
        self._broadcast_state()

    async def _event_loop(self) -> None:
        while True:
            event = await self._listener.get()
            self._consume_gateway_event(event.kind, event.payload)
            await self.force_refresh(refresh_services=False)

    async def _poll_loop(self) -> None:
        interval = max(float(self.config.health_poll_interval_sec), 0.2)
        while True:
            await asyncio.sleep(interval)
            await self.force_refresh(refresh_services=True)

    def _consume_gateway_event(self, kind: str, payload: dict[str, object]) -> None:
        if kind == "health" and str(payload.get("component", "")) == "aura_runtime":
            details = payload.get("details")
            if isinstance(details, dict) and isinstance(details.get("worldState"), dict):
                self._world_state = WorldStateSnapshot.from_dict(details["worldState"])
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
        frame = self.subscriber.current_frame
        dual_service = dict(self._service_state.get("dual", {}))
        services = {
            "navdp": dict(self._service_state.get("navdp", {})),
            "dual": dual_service,
            "system2": self._system2_service_state(
                session_active=request is not None,
                processes=processes,
                dual_service=dual_service,
            ),
        }
        transport_state = {
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
        return SnapshotAdapter.to_dashboard_state(
            self._world_state,
            processes=processes,
            services=services,
            session_state={
                "timestamp": time.time(),
                "active": request is not None,
                "startedAt": self.process_manager.session_started_at,
                "config": session_payload,
                "lastEvent": None if not self._event_logs else dict(self._event_logs[-1]),
            },
            transport_state=transport_state,
            architecture=self._build_architecture_state(
                session_active=request is not None,
                processes=processes,
                services=services,
                transport_state=transport_state,
            ),
            recent_logs=list(self._event_logs)[-20:],
            last_status=self._last_status,
            detector_capability=self._detector_capability,
        )

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
