"""State aggregation and runtime proxying for the backend."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import time
from typing import Any

from backend.app_keys import (
    API_BASE_URL,
    CONTROL_RUNTIME_URL,
    HTTP,
    INFERENCE_SYSTEM_URL,
    NAVIGATION_SYSTEM_URL,
    PLANNER_SYSTEM_URL,
    RUNTIME_OWNED,
    RUNTIME_SERVICE,
    RUNTIME_URL,
    WEBRTC_SERVICE,
)
from backend.models import DashboardStateBuilder, build_dashboard_catalog, parse_session_config
from backend.sources.control_runtime import fetch_runtime_status
from backend.sources.logs import merge_logs, tail_log
from backend.sources.navigation_system import fetch_navigation_status
from backend.sources.planner_system import fetch_planner_status
from backend.sources.runtime import fetch_runtime_state, post_runtime_session
from systems.shared.contracts.dashboard import LogRecord


def _event(message: str, *, level: str = "info") -> dict[str, object]:
    return LogRecord(
        source="backend",
        stream="event",
        level=level,
        message=message,
        timestampNs=int(time.time() * 1_000_000_000),
    ).to_dict()


def _process_index(processes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for process in processes:
        if not isinstance(process, dict):
            continue
        name = process.get("name")
        if isinstance(name, str) and name:
            indexed[name] = process
    return indexed


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _pixel_pair(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return [int(round(float(x))), int(round(float(y)))]
    return None


def _xyz_triplet(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        xyz = value[:3]
        if all(isinstance(item, (int, float)) for item in xyz):
            return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return None


def _overlay_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _selected_target_from_frame_meta(frame_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    payload = _as_dict(frame_meta)
    active_target = _as_dict(
        _overlay_value(payload, "activeTarget", "active_target")
    )
    if not active_target:
        detections = _as_list(payload.get("detections"))
        if detections:
            active_target = _as_dict(detections[0])
    if not active_target:
        return None

    summary: dict[str, Any] = {
        "className": str(
            active_target.get("className")
            or active_target.get("class_name")
            or active_target.get("label")
            or active_target.get("type")
            or "navigation_goal"
        ),
        "source": str(active_target.get("source") or "viewer"),
    }
    track_id = active_target.get("trackId", active_target.get("track_id"))
    if isinstance(track_id, str) and track_id:
        summary["trackId"] = track_id
    confidence = active_target.get("confidence")
    if isinstance(confidence, (int, float)):
        summary["confidence"] = float(confidence)
    depth_m = active_target.get("depthM", active_target.get("depth_m"))
    if isinstance(depth_m, (int, float)):
        summary["depthM"] = float(depth_m)
    nav_goal_pixel = _pixel_pair(
        active_target.get("navGoalPixel", active_target.get("nav_goal_pixel"))
    )
    if nav_goal_pixel is not None:
        summary["navGoalPixel"] = nav_goal_pixel
    bbox_xyxy = active_target.get("bboxXyxy", active_target.get("bbox_xyxy"))
    if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) == 4 and all(
        isinstance(item, (int, float)) for item in bbox_xyxy
    ):
        summary["bboxXyxy"] = [int(round(float(item))) for item in bbox_xyxy]
    world_pose_xyz = _xyz_triplet(
        active_target.get("worldPoseXyz", active_target.get("world_pose_xyz"))
    )
    if world_pose_xyz is not None:
        summary["worldPoseXyz"] = world_pose_xyz
    return summary


def _selected_target_from_status(
    runtime_status: dict[str, Any] | None,
    planner_status: dict[str, Any] | None,
    navigation_status: dict[str, Any] | None,
) -> dict[str, Any] | None:
    runtime_payload = _as_dict(runtime_status)
    planner_payload = _as_dict(planner_status)
    navigation_payload = _as_dict(navigation_status)
    nav_goal_pixel = _pixel_pair(
        _overlay_value(
            runtime_payload,
            "active_pixel_goal_xy",
            "goal_pixel_xy",
            "pending_pixel_goal_xy",
        )
    )
    system2_pixel_goal = _pixel_pair(
        _overlay_value(
            runtime_payload,
            "system2_pixel_goal",
            "system2PixelGoal",
        )
    )
    world_goal = _xyz_triplet(navigation_payload.get("goal_world_xy"))
    if world_goal is None:
        goal_world_xy = navigation_payload.get("goal_world_xy")
        if isinstance(goal_world_xy, (list, tuple)) and len(goal_world_xy) >= 2 and all(
            isinstance(item, (int, float)) for item in goal_world_xy[:2]
        ):
            world_goal = [float(goal_world_xy[0]), float(goal_world_xy[1]), 0.0]
    if nav_goal_pixel is None and system2_pixel_goal is None and world_goal is None:
        return None

    current_subgoal = _as_dict(planner_payload.get("current_subgoal"))
    label = (
        current_subgoal.get("label")
        or current_subgoal.get("target")
        or navigation_payload.get("instruction")
        or "Navigation Goal"
    )
    summary: dict[str, Any] = {
        "className": str(label),
        "source": "navigation",
    }
    if nav_goal_pixel is not None:
        summary["navGoalPixel"] = nav_goal_pixel
    elif system2_pixel_goal is not None:
        summary["navGoalPixel"] = system2_pixel_goal
    if world_goal is not None:
        summary["worldPoseXyz"] = world_goal
    return summary


def _server_service_snapshot(
    *,
    name: str,
    health_url: str,
    probe_result: dict[str, Any] | None,
    process: dict[str, Any] | None,
    session_active: bool,
    default_status: str = "inactive",
    extra_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ok = bool(probe_result and probe_result.get("ok"))
    process_state = str(process.get("state")) if isinstance(process, dict) else ""
    if ok:
        status = "healthy"
    elif process_state == "running":
        status = "degraded"
    elif session_active:
        status = "degraded"
    else:
        status = default_status

    health: dict[str, Any] = {}
    if extra_health:
        health.update(extra_health)
    if probe_result and probe_result.get("ok"):
        probe_payload = probe_result.get("status")
        if not isinstance(probe_payload, dict):
            probe_payload = probe_result.get("state")
        if isinstance(probe_payload, dict):
            health["probe"] = probe_payload
    elif probe_result and probe_result.get("error"):
        health["error"] = probe_result.get("error")
    if process is not None:
        health["process"] = dict(process)

    return {
        "name": name,
        "status": status,
        "healthUrl": health_url,
        "health": health,
    }


class DashboardSessionManager:
    """Build dashboard state and proxy lifecycle operations to the runtime service."""

    def __init__(self, app):
        self.app = app
        self.state_builder = DashboardStateBuilder(api_base_url=str(app[API_BASE_URL]).rstrip("/"))
        self.session_config: dict[str, Any] | None = None
        self._session_config_epoch = 0
        self.last_event = self.state_builder.default_state()["session"]["lastEvent"]
        self._state_cache: dict[str, Any] | None = None
        self._state_cache_expires_at = 0.0
        self._probe_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._state_cache_ttl_s = 0.25
        self._probe_cache_ttl_s = 1.0

    def _invalidate_caches(self) -> None:
        self._state_cache = None
        self._state_cache_expires_at = 0.0
        self._probe_cache.clear()

    def _set_session_config(self, config: dict[str, Any] | None) -> None:
        self._session_config_epoch += 1
        self.session_config = None if config is None else dict(config)
        self._invalidate_caches()

    def record_event(self, message: str, *, level: str = "info") -> None:
        self.last_event = _event(message, level=level)
        self._state_cache = None
        self._state_cache_expires_at = 0.0

    def _effective_session_config(
        self,
        runtime_session: dict[str, Any],
        *,
        observed_epoch: int,
    ) -> dict[str, Any] | None:
        runtime_config = runtime_session.get("config") if isinstance(runtime_session.get("config"), dict) else None
        session_active = bool(runtime_session.get("active"))
        if observed_epoch == self._session_config_epoch:
            if runtime_config is not None:
                self.session_config = dict(runtime_config)
            elif not session_active:
                self.session_config = None
        if runtime_config is not None:
            return dict(runtime_config)
        if session_active and self.session_config is not None:
            return dict(self.session_config)
        return None

    def _owned_runtime_service(self):
        service = self.app[RUNTIME_SERVICE]
        if service is None:
            raise RuntimeError("backend-owned runtime service is not configured")
        return service

    async def start_session(self, config: dict[str, Any]) -> dict[str, Any]:
        normalized_config = parse_session_config(config)
        if self.app[RUNTIME_OWNED]:
            response = await asyncio.to_thread(self._owned_runtime_service().start_session, dict(normalized_config))
        else:
            response = await post_runtime_session(
                self.app[HTTP],
                self.app[RUNTIME_URL],
                "/session/start",
                normalized_config,
                timeout_s=120.0,
            )
        if response.get("ok"):
            self._set_session_config(dict(normalized_config))
            return await self.build_state(force_refresh=True)
        self.record_event(str(response.get("error") or response.get("lastError") or "runtime start failed"), level="error")
        self._set_session_config(None)
        return await self.build_state(force_refresh=True)

    async def stop_session(self) -> dict[str, Any]:
        if self.app[RUNTIME_OWNED]:
            response = await asyncio.to_thread(self._owned_runtime_service().stop_session)
        else:
            response = await post_runtime_session(
                self.app[HTTP],
                self.app[RUNTIME_URL],
                "/session/stop",
                {},
                timeout_s=30.0,
            )
        if response.get("ok"):
            self._set_session_config(None)
            return await self.build_state(force_refresh=True)
        self.record_event(str(response.get("error") or response.get("lastError") or "runtime stop failed"), level="error")
        return await self.build_state(force_refresh=True)

    async def _cached_probe(
        self,
        cache_key: str,
        fetcher,
        *args,
        ttl_s: float | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        if not force_refresh:
            cached = self._probe_cache.get(cache_key)
            now = time.monotonic()
            if cached is not None and now < float(cached[0]):
                return deepcopy(cached[1])
        result = await asyncio.to_thread(fetcher, *args)
        payload = result if isinstance(result, dict) else {"ok": False, "error": "invalid_probe_payload"}
        expiry = time.monotonic() + (self._probe_cache_ttl_s if ttl_s is None else max(float(ttl_s), 0.0))
        self._probe_cache[cache_key] = (expiry, deepcopy(payload))
        return payload

    async def build_state(self, *, force_refresh: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if not force_refresh and self._state_cache is not None and now < self._state_cache_expires_at:
            return deepcopy(self._state_cache)
        state = self.state_builder.default_state()
        state["timestamp"] = time.time()
        state["session"]["lastEvent"] = self.last_event
        session_config_epoch = self._session_config_epoch
        webrtc_service = self.app[WEBRTC_SERVICE]
        webrtc_health = None if webrtc_service is None else webrtc_service.health_snapshot()
        latest_frame_state = None
        latest_frame_meta = None
        if webrtc_service is not None:
            subscriber = getattr(webrtc_service, "subscriber", None)
            if subscriber is not None:
                latest_frame_state = subscriber.build_state_snapshot()
                latest_frame_meta = subscriber.build_frame_meta()
        if isinstance(webrtc_health, dict):
            transport_health = webrtc_health.get("transportHealth")
            if not isinstance(transport_health, dict):
                transport_health = {}
            state["transport"].update(
                {
                    "transport": webrtc_health.get("transport"),
                    "transportHealth": transport_health,
                    "busHealth": transport_health,
                    "mediaIngress": webrtc_health.get("mediaIngress"),
                    "mediaEgress": webrtc_health.get("mediaEgress"),
                }
            )

        if self.app[RUNTIME_OWNED]:
            try:
                runtime_service_result = {"ok": True, "state": await asyncio.to_thread(self._owned_runtime_service().state_payload)}
            except Exception as exc:  # noqa: BLE001
                runtime_service_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            runtime_service_result = await fetch_runtime_state(self.app[HTTP], self.app[RUNTIME_URL])

        runtime_result, planner_result, navigation_result = await asyncio.gather(
            self._cached_probe(
                "control_runtime_status",
                fetch_runtime_status,
                self.app[CONTROL_RUNTIME_URL],
                force_refresh=force_refresh,
            ),
            self._cached_probe(
                "planner_status",
                fetch_planner_status,
                self.app[PLANNER_SYSTEM_URL],
                force_refresh=force_refresh,
            ),
            self._cached_probe(
                "navigation_status",
                fetch_navigation_status,
                self.app[NAVIGATION_SYSTEM_URL],
                force_refresh=force_refresh,
            ),
        )
        inference_result: dict[str, Any] = {"ok": False, "error": "navigation-owned backends"}
        runtime_status_payload: dict[str, Any] | None = None
        planner_status_payload: dict[str, Any] | None = None
        navigation_status_payload: dict[str, Any] | None = None

        state["services"]["runtime"] = {
            "name": "runtime",
            "status": "healthy" if runtime_service_result.get("ok") else "degraded",
            "healthUrl": None if self.app[RUNTIME_OWNED] else f"{self.app[RUNTIME_URL]}/session/state",
            "health": (
                {
                    **(runtime_service_result.get("state") if runtime_service_result.get("ok") else {"error": runtime_service_result.get("error")}),
                    "ownedByBackend": bool(self.app[RUNTIME_OWNED]),
                }
            ),
        }

        if runtime_service_result.get("ok"):
            runtime_service_state = runtime_service_result.get("state") or {}
            runtime_session = runtime_service_state.get("session") if isinstance(runtime_service_state.get("session"), dict) else {}
            effective_session_config = self._effective_session_config(runtime_session, observed_epoch=session_config_epoch)
            state["session"] = {
                "active": bool(runtime_session.get("active")),
                "startedAt": runtime_session.get("startedAt"),
                "config": effective_session_config,
                "lastEvent": runtime_session.get("lastEvent") or self.last_event,
            }
            self.last_event = state["session"]["lastEvent"]
            processes = runtime_service_state.get("processes")
            state["processes"] = list(processes) if isinstance(processes, list) else []
            state["transport"]["viewerEnabled"] = bool((effective_session_config or {}).get("viewerEnabled"))
        else:
            fallback_active = bool(runtime_result.get("ok"))
            reason = str(runtime_service_result.get("error") or "runtime unavailable")
            self.record_event(reason, level="warning")
            state["session"] = {
                "active": fallback_active,
                "startedAt": None,
                "config": None if not fallback_active else (None if self.session_config is None else dict(self.session_config)),
                "lastEvent": self.last_event,
            }
            state["processes"] = []

        configured_viewer_enabled = bool(_as_dict(state["session"].get("config")).get("viewerEnabled"))
        if isinstance(webrtc_health, dict):
            transport_mode = "disabled" if not configured_viewer_enabled else str(webrtc_health.get("transport") or "")
            frame_available = bool(webrtc_health.get("frameAvailable"))
            state["transport"].update(
                {
                    "viewerEnabled": configured_viewer_enabled,
                    "frameAgeMs": webrtc_health.get("frameAgeMs"),
                    "lastGoodFrameAgeMs": webrtc_health.get("lastGoodFrameAgeMs"),
                    "frameSeq": webrtc_health.get("frameSeq"),
                    "frameAvailable": frame_available if configured_viewer_enabled else False,
                    "streamStalled": bool(webrtc_health.get("streamStalled")) if configured_viewer_enabled else False,
                    "dropCounters": webrtc_health.get("dropCounters") if configured_viewer_enabled else {"shmOverwrite": 0},
                    "peerActive": bool(webrtc_health.get("peerActive")),
                    "peerSessionId": webrtc_health.get("peerSessionId"),
                    "peerTrackRoles": list(webrtc_health.get("peerTrackRoles", []))
                    if isinstance(webrtc_health.get("peerTrackRoles"), list)
                    else [],
                    "transport": transport_mode,
                    "transportHealth": webrtc_health.get("transportHealth"),
                    "busHealth": webrtc_health.get("transportHealth"),
                    "mediaIngress": "disabled" if not configured_viewer_enabled else webrtc_health.get("mediaIngress"),
                    "mediaEgress": "disabled" if not configured_viewer_enabled else webrtc_health.get("mediaEgress"),
                }
            )
            state["latencyBreakdown"]["frameAgeMs"] = webrtc_health.get("frameAgeMs")
            state["sensors"]["rgbAvailable"] = bool(webrtc_health.get("rgbAvailable", frame_available))
            state["sensors"]["depthAvailable"] = bool(webrtc_health.get("depthAvailable"))
            state["sensors"]["source"] = webrtc_health.get("source", state["sensors"]["source"])
            state["sensors"]["frameId"] = webrtc_health.get("frameId")
            if not configured_viewer_enabled:
                state["architecture"]["modules"]["telemetry"]["status"] = "inactive"
                state["architecture"]["modules"]["telemetry"]["summary"] = "Viewer publish disabled"
                state["architecture"]["modules"]["telemetry"]["detail"] = "viewer publish disabled"
            elif frame_available:
                state["architecture"]["modules"]["telemetry"]["status"] = "healthy"
                state["architecture"]["modules"]["telemetry"]["summary"] = "WebRTC live viewer"
                state["architecture"]["modules"]["telemetry"]["detail"] = str(
                    webrtc_health.get("source") or "control_runtime"
                )
            else:
                state["architecture"]["modules"]["telemetry"]["status"] = "inactive"
                state["architecture"]["modules"]["telemetry"]["summary"] = "Waiting for WebRTC frames"
                state["architecture"]["modules"]["telemetry"]["detail"] = str(
                    webrtc_health.get("source") or "viewer unavailable"
                )

        if runtime_result.get("ok"):
            runtime_status = dict(runtime_result["status"])
            runtime_status_payload = dict(runtime_status)
            viewer_status = runtime_status.get("viewer") if isinstance(runtime_status.get("viewer"), dict) else {}
            state["runtime"].update(runtime_status)
            effective_viewer_status = dict(viewer_status)
            if webrtc_health is not None:
                effective_viewer_status.update(webrtc_health)
            frame_age_ms = effective_viewer_status.get("frameAgeMs")
            frame_available = bool(effective_viewer_status.get("frameAvailable"))
            transport_mode = "disabled" if not configured_viewer_enabled else str(effective_viewer_status.get("transport") or "")
            state["transport"].update(
                {
                    "viewerEnabled": configured_viewer_enabled,
                    "frameAgeMs": frame_age_ms,
                    "lastGoodFrameAgeMs": effective_viewer_status.get("lastGoodFrameAgeMs"),
                    "frameSeq": effective_viewer_status.get("frameSeq"),
                    "frameAvailable": frame_available,
                    "streamStalled": bool(effective_viewer_status.get("streamStalled")) if configured_viewer_enabled else False,
                    "dropCounters": effective_viewer_status.get("dropCounters")
                    if configured_viewer_enabled
                    else {"shmOverwrite": 0},
                    "peerActive": bool(effective_viewer_status.get("peerActive")),
                    "peerSessionId": effective_viewer_status.get("peerSessionId"),
                    "peerTrackRoles": list(effective_viewer_status.get("peerTrackRoles", []))
                    if isinstance(effective_viewer_status.get("peerTrackRoles"), list)
                    else [],
                    "transport": transport_mode,
                    "transportHealth": effective_viewer_status.get("transportHealth"),
                    "busHealth": effective_viewer_status.get("transportHealth"),
                    "mediaIngress": "disabled" if not configured_viewer_enabled else effective_viewer_status.get("mediaIngress"),
                    "mediaEgress": "disabled" if not configured_viewer_enabled else effective_viewer_status.get("mediaEgress"),
                }
            )
            state["latencyBreakdown"]["frameAgeMs"] = frame_age_ms
            state["sensors"]["rgbAvailable"] = bool(effective_viewer_status.get("rgbAvailable", frame_available))
            state["sensors"]["depthAvailable"] = bool(effective_viewer_status.get("depthAvailable"))
            state["sensors"]["poseAvailable"] = True
            state["sensors"]["source"] = effective_viewer_status.get("source", state["sensors"]["source"])
            state["sensors"]["frameId"] = effective_viewer_status.get("frameId")
            state["architecture"]["mainControlServer"]["status"] = "healthy"
            state["architecture"]["mainControlServer"]["summary"] = runtime_status.get("state_label", "idle")
            state["architecture"]["mainControlServer"]["detail"] = runtime_status.get("navigation_status", "")
            state["architecture"]["modules"]["locomotion"]["status"] = runtime_status.get("state_label", "unknown")
            state["architecture"]["modules"]["locomotion"]["summary"] = "Local trajectory follower"
            state["architecture"]["modules"]["locomotion"]["detail"] = runtime_status.get("navigation_status", "")
            if not configured_viewer_enabled:
                state["architecture"]["modules"]["telemetry"]["status"] = "inactive"
                state["architecture"]["modules"]["telemetry"]["summary"] = "Viewer publish disabled"
                state["architecture"]["modules"]["telemetry"]["detail"] = "viewer publish disabled"
            else:
                state["architecture"]["modules"]["telemetry"]["status"] = "healthy" if frame_available else "inactive"
                state["architecture"]["modules"]["telemetry"]["summary"] = (
                    "WebRTC live viewer" if frame_available else "Waiting for WebRTC frames"
                )
                state["architecture"]["modules"]["telemetry"]["detail"] = effective_viewer_status.get(
                    "source",
                    "viewer unavailable",
                )
        else:
            state["runtime"]["lastStatusEvent"] = {"state": "unreachable", "reason": runtime_result.get("error")}
            state["transport"]["viewerEnabled"] = bool((self.session_config or {}).get("viewerEnabled"))

        if planner_result.get("ok"):
            planner_status = dict(planner_result["status"])
            planner_status_payload = dict(planner_status)
            state["runtime"]["plannerControlMode"] = planner_status.get("task_status", "idle")
            state["runtime"]["activeInstruction"] = planner_status.get("instruction") or ""
            state["runtime"]["taskId"] = planner_status.get("task_id")
            state["runtime"]["taskFrame"] = planner_status.get("task_frame")
            state["runtime"]["currentSubgoal"] = planner_status.get("current_subgoal")
            state["runtime"]["subgoals"] = planner_status.get("subgoals", [])
            state["runtime"]["lastStatusEvent"] = {
                "state": planner_status.get("task_status", "idle"),
                "reason": planner_status.get("last_error"),
            }
            if state["runtime"].get("executionMode") == "IDLE":
                state["runtime"]["executionMode"] = "NAV" if planner_status.get("task_status") == "running" else "IDLE"
            state["services"]["planner"] = {
                "name": "planner",
                "status": "healthy" if planner_status.get("ok", True) else "degraded",
                "healthUrl": f"{self.app[PLANNER_SYSTEM_URL]}/planner/status",
                "health": planner_status,
            }
            state["services"]["dual"] = dict(state["services"]["planner"])
            state["architecture"]["mainControlServer"]["core"]["plannerCoordinator"]["status"] = "healthy"
            state["architecture"]["mainControlServer"]["core"]["plannerCoordinator"]["summary"] = planner_status.get("task_status", "idle")
            current_subgoal = planner_status.get("current_subgoal") or {}
            state["architecture"]["mainControlServer"]["core"]["plannerCoordinator"]["detail"] = str(
                current_subgoal.get("type") or planner_status.get("instruction") or ""
            )
        else:
            planner_error = planner_result.get("error")
            state["services"]["planner"]["status"] = "degraded"
            state["services"]["planner"]["health"] = {"error": planner_error}
            state["services"]["dual"] = dict(state["services"]["planner"])

        if navigation_result.get("ok"):
            navigation_status = dict(navigation_result["status"])
            navigation_status_payload = dict(navigation_status)
            state["architecture"]["modules"]["nav"]["status"] = navigation_status.get("status", "unknown")
            state["architecture"]["modules"]["nav"]["summary"] = f"path pts {navigation_status.get('path_points', 0)}"
            state["architecture"]["modules"]["nav"]["detail"] = navigation_status.get("instruction") or ""
            state["runtime"]["routeState"] = {
                "pathPoints": navigation_status.get("path_points", 0),
                "trajectoryAgeSec": navigation_status.get("plan_age_s"),
            }
        else:
            state["architecture"]["modules"]["nav"]["status"] = "degraded"
            state["architecture"]["modules"]["nav"]["detail"] = str(navigation_result.get("error"))

        if navigation_status_payload is not None:
            system2 = _as_dict(navigation_status_payload.get("system2"))
            system2_stage = _as_dict(navigation_status_payload.get("system2_stage"))
            navdp = _as_dict(navigation_status_payload.get("system1") or navigation_status_payload.get("navdp"))
            state["services"]["navdp"] = {
                "name": "navdp",
                "status": navdp.get("status", "unknown"),
                "healthUrl": navdp.get("health_url"),
                "latencyMs": navdp.get("latency_ms"),
                "health": navdp,
            }
            state["services"]["system2"] = {
                "name": "system2",
                "status": system2_stage.get("status", system2.get("status", "unknown")),
                "healthUrl": system2_stage.get("health_url"),
                "latencyMs": system2.get("latency_ms") or system2_stage.get("latency_ms"),
                "health": system2_stage,
                "output": system2,
            }
            state["architecture"]["gateway"]["status"] = "healthy"
            state["architecture"]["gateway"]["summary"] = "Navigation backends"
            state["architecture"]["gateway"]["detail"] = "navigation-owned system2 and system1 backends"
            state["architecture"]["modules"]["s2"]["status"] = system2_stage.get("status", system2.get("status", "unknown"))
            state["architecture"]["modules"]["s2"]["latencyMs"] = system2.get("latency_ms") or system2_stage.get("latency_ms")
        else:
            state["architecture"]["gateway"]["status"] = "degraded"
            state["architecture"]["gateway"]["summary"] = "Navigation backends"
            state["architecture"]["gateway"]["detail"] = str(navigation_result.get("error"))

        session_active = bool(state["session"].get("active"))
        process_index = _process_index(state["processes"])
        state["services"]["backend"] = {
            "name": "backend",
            "status": "healthy",
            "healthUrl": f"{self.app[API_BASE_URL]}/api/state",
            "health": {
                "ownedRuntime": bool(self.app[RUNTIME_OWNED]),
                "webrtcEmbedded": webrtc_service is not None,
            },
        }
        state["services"]["controlRuntime"] = _server_service_snapshot(
            name="control_runtime",
            health_url=f"{self.app[CONTROL_RUNTIME_URL]}/runtime/status",
            probe_result=runtime_result,
            process=process_index.get("control_runtime"),
            session_active=session_active,
        )
        state["services"]["inferenceSystem"] = _server_service_snapshot(
            name="navigation_backends",
            health_url=f"{self.app[NAVIGATION_SYSTEM_URL]}/navigation/status",
            probe_result=navigation_result,
            process=process_index.get("navigation_system"),
            session_active=session_active,
        )
        state["services"]["navigationSystem"] = _server_service_snapshot(
            name="navigation_system",
            health_url=f"{self.app[NAVIGATION_SYSTEM_URL]}/navigation/status",
            probe_result=navigation_result,
            process=process_index.get("navigation_system"),
            session_active=session_active,
        )
        state["services"]["plannerSystem"] = _server_service_snapshot(
            name="planner_system",
            health_url=f"{self.app[PLANNER_SYSTEM_URL]}/planner/status",
            probe_result=planner_result,
            process=process_index.get("planner_system"),
            session_active=session_active,
        )

        frame_meta_payload = _as_dict(latest_frame_meta)
        if frame_meta_payload:
            detections = _as_list(frame_meta_payload.get("detections"))
            trajectory_pixels = _as_list(_overlay_value(frame_meta_payload, "trajectoryPixels", "trajectory_pixels"))
            state["perception"]["detectionCount"] = len(detections)
            state["perception"]["trackedDetectionCount"] = len(detections)
            state["perception"]["trajectoryPointCount"] = len(trajectory_pixels)

        selected_target_summary = _selected_target_from_frame_meta(frame_meta_payload)
        if selected_target_summary is None:
            selected_target_summary = _selected_target_from_status(
                runtime_status_payload,
                planner_status_payload,
                navigation_status_payload,
            )
        state["selectedTargetSummary"] = selected_target_summary

        log_groups: list[list[dict[str, object]]] = []
        for process in state["processes"]:
            stdout_log = process.get("stdoutLog")
            stderr_log = process.get("stderrLog")
            if isinstance(stdout_log, str):
                log_groups.append(tail_log(stdout_log, source=str(process["name"]), stream="stdout"))
            if isinstance(stderr_log, str):
                log_groups.append(tail_log(stderr_log, source=str(process["name"]), stream="stderr"))
        session_event = state["session"].get("lastEvent")
        state["logs"] = merge_logs(*log_groups, [session_event] if isinstance(session_event, dict) else [], limit=80)
        if latest_frame_state is not None:
            state["transport"]["latestFrameState"] = latest_frame_state
        if latest_frame_meta is not None:
            state["transport"]["latestFrameMeta"] = latest_frame_meta
        state["dashboardCatalog"] = build_dashboard_catalog(state)
        self._state_cache = deepcopy(state)
        self._state_cache_expires_at = time.monotonic() + self._state_cache_ttl_s
        return state
