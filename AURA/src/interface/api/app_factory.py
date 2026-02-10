import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Set, Tuple

sys.path.append(str(Path(__file__).resolve().parents[3]))

from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.application.streaming.perception_loop import run_perception_stream
from src.application.streaming.runtime_manager import RuntimeManager
from src.infrastructure.streaming.config import StreamServerConfig
from src.infrastructure.streaming.frame_sources import (
    Ros2ImageTopicFrameSource,
    VideoFileFrameSource,
)
from src.infrastructure.streaming.websocket_sinks import WebSocketVideoSink, WebSocketMetadataSink
from src.infrastructure.streaming.webrtc_sinks import (
    PerceptionVideoTrack,
    WebRtcVideoSink,
    WebRtcMetadataSink,
)


def create_app() -> FastAPI:
    cfg = StreamServerConfig()
    runtime = RuntimeManager(cfg)
    pcs: Set[RTCPeerConnection] = set()
    dashboard_build_dir = cfg.root_dir.parent / "dashboard" / "build"
    dashboard_assets_dir = dashboard_build_dir / "assets"
    pipeline_script_path = cfg.root_dir.parent / "run_pipeline.sh"
    pipeline_log_path = Path(
        os.getenv("AURA_MANAGED_PIPELINE_LOG", "/tmp/aura_managed_pipeline.log")
    )
    managed_pipeline_proc: Optional[subprocess.Popen] = None
    managed_pipeline_started_at: Optional[float] = None
    managed_pipeline_last_exit_code: Optional[int] = None
    managed_pipeline_lock = asyncio.Lock()
    habitat_control_host = os.getenv("HABITAT_CONTROL_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        habitat_control_port = int(os.getenv("HABITAT_CONTROL_PORT", "8766"))
    except ValueError:
        habitat_control_port = 8766
    habitat_supported_actions = (
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "move_up",
        "move_down",
        "turn_left",
        "turn_right",
        "look_up",
        "look_down",
    )

    app = FastAPI(title="Perception Streamer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if dashboard_build_dir.exists():
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(dashboard_build_dir), html=True),
            name="dashboard",
        )
    if dashboard_assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(dashboard_assets_dir), html=False),
            name="dashboard-assets",
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            await _close_peers(pcs)
            runtime.shutdown()

    app.router.lifespan_context = lifespan

    def _refresh_managed_pipeline_state() -> None:
        nonlocal managed_pipeline_proc, managed_pipeline_started_at, managed_pipeline_last_exit_code
        if managed_pipeline_proc is None:
            return
        return_code = managed_pipeline_proc.poll()
        if return_code is None:
            return
        managed_pipeline_last_exit_code = return_code
        managed_pipeline_proc = None
        managed_pipeline_started_at = None

    def _managed_pipeline_status() -> Dict[str, Any]:
        _refresh_managed_pipeline_state()
        is_running = managed_pipeline_proc is not None
        now = time.time()
        return {
            "running": is_running,
            "pid": managed_pipeline_proc.pid if managed_pipeline_proc else None,
            "startedAt": managed_pipeline_started_at,
            "uptimeSec": (
                max(0.0, now - managed_pipeline_started_at)
                if managed_pipeline_started_at is not None and is_running
                else None
            ),
            "lastExitCode": managed_pipeline_last_exit_code,
            "logPath": str(pipeline_log_path),
        }

    def _bool_option(
        options: Dict[str, Any], option_key: str, default: bool
    ) -> bool:
        raw = options.get(option_key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return raw != 0
        if isinstance(raw, str):
            return raw.strip().lower() not in {"0", "false", "no", "off", ""}
        return default

    def _build_pipeline_env(overrides: Dict[str, Any]) -> Dict[str, str]:
        env = os.environ.copy()

        # GUI server already serves dashboard and stream API, so disable
        # duplicated UI/server launch in the spawned pipeline script.
        env["DASHBOARD_AUTO_OPEN"] = "0"
        env["AURA_ENABLE_STREAM_SERVER"] = "0"

        bool_mapping = (
            ("enableSemantic", "ENABLE_SEMANTIC", True),
            ("enableOctomap", "ENABLE_OCTOMAP", True),
            ("enableLlm", "ENABLE_LLM", False),
            ("useNav2", "USE_NAV2", True),
            ("nav2AutoExplore", "NAV2_AUTO_EXPLORE", True),
            ("habitatManualOnly", "HABITAT_MANUAL_ONLY", False),
            ("habitatAddPlayerAgent", "HABITAT_ADD_PLAYER_AGENT", False),
        )
        for option_key, env_key, default in bool_mapping:
            env[env_key] = "1" if _bool_option(overrides, option_key, default) else "0"

        yolo_model = overrides.get("yoloModelPath")
        if isinstance(yolo_model, str) and yolo_model.strip():
            env["YOLO_MODEL"] = yolo_model.strip()

        habitat_dataset = overrides.get("habitatDataset")
        if isinstance(habitat_dataset, str) and habitat_dataset.strip():
            env["HABITAT_SCENE_DATASET"] = habitat_dataset.strip()

        octomap_resolution = overrides.get("octomapResolution")
        if octomap_resolution is not None:
            env["OCTOMAP_RESOLUTION"] = str(octomap_resolution)

        return env

    @app.get("/")
    async def root():
        return JSONResponse(
            {
                "status": "ok",
                "message": "WebRTC offer at /webrtc/offer (metadata datachannel + video track). WebSocket legacy at /ws/stream",
                "inputSource": cfg.input_source,
                "videoPath": str(cfg.video_path) if cfg.input_source == "video" else None,
                "rosImageTopic": cfg.ros_image_topic if cfg.input_source == "ros2" else None,
                "rosSlamPoseTopic": cfg.ros_slam_pose_topic if cfg.input_source == "ros2" else None,
                "rosSemanticProjectedMapTopic": (
                    cfg.ros_semantic_projected_map_topic if cfg.input_source == "ros2" else None
                ),
                "rosSemanticOctomapCloudTopic": (
                    cfg.ros_semantic_octomap_cloud_topic if cfg.input_source == "ros2" else None
                ),
                "targetFps": cfg.target_fps,
                "frameMaxWidth": cfg.frame_max_width,
                "jpegQuality": cfg.jpeg_quality,
                "llmModel": cfg.llm_model_name,
                "llmDevice": cfg.llm_device,
                "runtimeReady": runtime.perception is not None,
                "webrtcOfferPath": "/webrtc/offer",
                "webrtcDataChannel": "metadata",
                "dashboardPath": "/dashboard" if dashboard_build_dir.exists() else None,
            }
        )

    @app.post("/system/shutdown")
    async def shutdown_pipeline():
        async with managed_pipeline_lock:
            status = _managed_pipeline_status()

        shutdown_target: Optional[str] = None
        shutdown_target_pid: Optional[int] = None

        if status["running"] and status["pid"] is not None:
            shutdown_target_pid = int(status["pid"])
            shutdown_target = "managed-pipeline"
        else:
            external_target_pid = _resolve_external_shutdown_target_pid()
            if external_target_pid is not None:
                shutdown_target_pid = int(external_target_pid)
                shutdown_target = "external-controller"

        if shutdown_target_pid is None or shutdown_target is None:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "idle",
                    "message": "No running managed pipeline found.",
                },
            )

        termination = await _terminate_pid_tree(shutdown_target_pid, graceful_timeout_sec=8.0)
        if shutdown_target == "managed-pipeline":
            async with managed_pipeline_lock:
                status = _managed_pipeline_status()

        if not termination["terminated"]:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to fully terminate pipeline target.",
                    "targetPid": shutdown_target_pid,
                    "target": shutdown_target,
                    "termination": termination,
                    "pipeline": status,
                },
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": "Pipeline target terminated.",
                "targetPid": shutdown_target_pid,
                "target": shutdown_target,
                "termination": termination,
                "pipeline": status,
            },
        )

    @app.get("/system/pipeline/status")
    async def pipeline_status():
        async with managed_pipeline_lock:
            status = _managed_pipeline_status()
        return JSONResponse(
            {
                "status": "ok",
                "pipeline": status,
                "scriptPath": str(pipeline_script_path),
            }
        )

    @app.post("/system/pipeline/start")
    async def start_pipeline(payload: Optional[Dict[str, Any]] = None):
        nonlocal managed_pipeline_proc, managed_pipeline_started_at, managed_pipeline_last_exit_code
        if not pipeline_script_path.exists():
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Pipeline script not found: {pipeline_script_path}",
                },
            )

        if not os.access(pipeline_script_path, os.X_OK):
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Pipeline script is not executable: {pipeline_script_path}",
                },
            )

        overrides: Dict[str, Any] = payload or {}

        async with managed_pipeline_lock:
            existing_status = _managed_pipeline_status()
            if existing_status["running"]:
                return JSONResponse(
                    status_code=409,
                    content={
                        "status": "already-running",
                        "message": "Pipeline is already running.",
                        "pipeline": existing_status,
                    },
                )

            env = _build_pipeline_env(overrides)
            try:
                log_file = open(pipeline_log_path, "ab")
            except OSError as exc:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": f"Failed to open pipeline log file: {exc}",
                    },
                )

            try:
                process = subprocess.Popen(
                    [str(pipeline_script_path)],
                    cwd=str(cfg.root_dir.parent),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as exc:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": f"Failed to start pipeline: {exc}",
                    },
                )
            finally:
                log_file.close()

            managed_pipeline_proc = process
            managed_pipeline_started_at = time.time()
            managed_pipeline_last_exit_code = None
            started_status = _managed_pipeline_status()

        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "message": "Pipeline start requested.",
                "pipeline": started_status,
                "scriptPath": str(pipeline_script_path),
            },
        )

    @app.post("/system/habitat/action")
    async def send_habitat_action(payload: Optional[Dict[str, Any]] = None):
        if habitat_control_port <= 0:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "disabled",
                    "message": "Habitat control channel is disabled (HABITAT_CONTROL_PORT <= 0).",
                    "host": habitat_control_host,
                    "port": habitat_control_port,
                },
            )

        request_data = payload or {}
        action = request_data.get("action")
        if not isinstance(action, str):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "invalid-request",
                    "message": "Missing action string.",
                    "supportedActions": habitat_supported_actions,
                },
            )
        action = action.strip()
        if action not in habitat_supported_actions:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "invalid-action",
                    "message": f"Unsupported action: {action}",
                    "supportedActions": habitat_supported_actions,
                },
            )

        repeat_raw = request_data.get("repeat", 1)
        try:
            repeat = int(repeat_raw)
        except (TypeError, ValueError):
            repeat = 1
        repeat = max(1, min(repeat, 10))

        packet = json.dumps({"action": action, "repeat": repeat}).encode("utf-8")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(packet, (habitat_control_host, habitat_control_port))
        except OSError as exc:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Failed to send action to Habitat: {exc}",
                    "host": habitat_control_host,
                    "port": habitat_control_port,
                },
            )

        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "action": action,
                "repeat": repeat,
                "target": f"udp://{habitat_control_host}:{habitat_control_port}",
            },
        )

    @app.post("/system/server/shutdown")
    async def shutdown_server_and_dashboard():
        async with managed_pipeline_lock:
            status = _managed_pipeline_status()

        shutdown_target: Optional[str] = None
        shutdown_target_pid: Optional[int] = None
        termination: Optional[Dict[str, Any]] = None

        if status["running"] and status["pid"] is not None:
            shutdown_target_pid = int(status["pid"])
            shutdown_target = "managed-pipeline"
        else:
            external_target_pid = _resolve_external_shutdown_target_pid()
            if external_target_pid is not None:
                shutdown_target_pid = int(external_target_pid)
                shutdown_target = "external-controller"

        if shutdown_target_pid is not None and shutdown_target is not None:
            termination = await _terminate_pid_tree(
                shutdown_target_pid,
                graceful_timeout_sec=8.0,
            )
            if shutdown_target == "managed-pipeline":
                async with managed_pipeline_lock:
                    status = _managed_pipeline_status()

            if not termination["terminated"]:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": "Failed to terminate pipeline before server shutdown.",
                        "pipelineRunning": bool(status.get("running", False)),
                        "target": shutdown_target,
                        "targetPid": shutdown_target_pid,
                        "termination": termination,
                    },
                )

        shutdown_message = (
            "Pipeline termination completed. Backend shutdown scheduled."
            if shutdown_target is not None
            else "Backend shutdown scheduled."
        )
        asyncio.create_task(_signal_pid_after_delay(os.getpid(), 0.35))
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "message": shutdown_message,
                "pipelineRunning": bool(status.get("running", False)),
                "target": shutdown_target,
                "targetPid": shutdown_target_pid,
                "termination": termination,
            },
        )

    @app.websocket("/ws/stream")
    async def stream_frames(websocket: WebSocket):
        await websocket.accept()

        if runtime.perception is None or runtime.scene_planner is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Runtime is not ready yet.",
                }
            )
            await websocket.close(code=1011)
            return

        await websocket.send_json(
            {
                "type": "init",
                "runtimeReady": True,
                "llmModel": cfg.llm_model_name,
                "device": cfg.llm_device,
            }
        )

        stop_event = asyncio.Event()
        video_sink = WebSocketVideoSink(websocket, cfg.frame_max_width, cfg.jpeg_quality)
        metadata_sink = WebSocketMetadataSink(websocket)
        use_source_timing = cfg.input_source == "ros2"

        try:
            frame_source = _create_frame_source(cfg)
        except Exception as exc:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Failed to initialize frame source: {exc}",
                }
            )
            await websocket.close(code=1011)
            return

        try:
            await run_perception_stream(
                runtime.perception,
                runtime.scene_planner,
                video_sink,
                metadata_sink,
                frame_source,
                cfg.target_fps,
                stop_event,
                use_source_timing=use_source_timing,
            )
        except WebSocketDisconnect:
            pass
        finally:
            stop_event.set()

    @app.post("/webrtc/offer")
    async def webrtc_offer(payload: Dict[str, Any]):
        if runtime.perception is None or runtime.scene_planner is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Runtime is not ready yet."},
            )

        pc = RTCPeerConnection(
            RTCConfiguration([RTCIceServer(urls=["stun:stun.l.google.com:19302"])]),
        )
        pcs.add(pc)

        frame_queue: "asyncio.Queue[Optional[Tuple[int, Any]]]" = asyncio.Queue(
            maxsize=1
        )
        stop_event = asyncio.Event()
        metadata_channel: Optional[RTCDataChannel] = None
        video_track = PerceptionVideoTrack(frame_queue, stop_event)

        @pc.on("datachannel")
        def on_datachannel(channel):
            nonlocal metadata_channel
            if channel.label == "metadata":
                metadata_channel = channel

                @channel.on("close")
                def _on_metadata_close():
                    stop_event.set()

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            state = pc.connectionState
            print(f"[WebRTC] PC State: {state}")

            if state in {"failed", "closed", "disconnected"}:
                stop_event.set()
                pcs.discard(pc)

                try:
                    await pc.close()
                except Exception:
                    pass

                # 🔥 모든 비동기 태스크도 정리
                for t in getattr(pc, "_tasks", []):
                    if not t.done():
                        t.cancel()

        remote_sdp = payload.get("sdp", "")
        remote_type = payload.get("type", "offer")
        await pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=remote_sdp,
                type=remote_type,
            )
        )

        pc.addTrack(video_track)

        remote_candidates = payload.get("iceCandidates") or payload.get("ice_candidates") or []
        for cand in remote_candidates:
            try:
                await pc.addIceCandidate(
                    RTCIceCandidate(
                        sdpMid=cand.get("sdpMid"),
                        sdpMLineIndex=int(cand.get("sdpMLineIndex", 0) or 0),
                        candidate=cand.get("candidate", ""),
                    )
                )
            except Exception:
                pass

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_gathering_complete(pc)
        local_candidates = _gather_local_candidates(pc)

        video_sink = WebRtcVideoSink(frame_queue, stop_event, cfg.frame_max_width)
        metadata_sink = WebRtcMetadataSink(lambda: metadata_channel)
        use_source_timing = cfg.input_source == "ros2"

        try:
            frame_source = _create_frame_source(cfg)
        except Exception as exc:
            stop_event.set()
            pcs.discard(pc)
            try:
                await pc.close()
            except Exception:
                pass
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to initialize frame source: {exc}"},
            )

        async def on_shutdown():
            if frame_queue.empty():
                await frame_queue.put(None)
            stop_event.set()

        asyncio.create_task(
            run_perception_stream(
                runtime.perception,
                runtime.scene_planner,
                video_sink,
                metadata_sink,
                frame_source,
                cfg.target_fps,
                stop_event,
                on_shutdown,
                use_source_timing=use_source_timing,
            )
        )

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "iceCandidates": local_candidates,
        }

    return app


async def _close_peers(pcs: Set[RTCPeerConnection]) -> None:
    for pc in list(pcs):
        try:
            await pc.close()
        except Exception:
            pass
        pcs.discard(pc)


async def _signal_pid_after_delay(pid: int, delay_sec: float = 0.2) -> None:
    await asyncio.sleep(delay_sec)
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _is_pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _resolve_terminatable_process_group(pid: int) -> Optional[int]:
    if not _is_pid_alive(pid):
        return None

    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None

    if pgid <= 1:
        return None

    # Never send a group signal to the current API server process group.
    try:
        if pgid == os.getpgrp():
            return None
    except Exception:
        return None
    return pgid


def _is_process_group_alive(pgid: int) -> bool:
    if pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _send_signal_to_target(
    pid: int,
    sig: signal.Signals,
    pgid: Optional[int],
) -> bool:
    try:
        if pgid is not None:
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


async def _wait_for_target_exit(
    pid: int,
    pgid: Optional[int],
    timeout_sec: float,
    poll_interval_sec: float = 0.2,
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while time.monotonic() < deadline:
        alive = _is_process_group_alive(pgid) if pgid is not None else _is_pid_alive(pid)
        if not alive:
            return True
        await asyncio.sleep(max(0.05, poll_interval_sec))

    alive = _is_process_group_alive(pgid) if pgid is not None else _is_pid_alive(pid)
    return not alive


async def _terminate_pid_tree(
    pid: int,
    graceful_timeout_sec: float = 8.0,
    force_timeout_sec: float = 2.0,
) -> Dict[str, Any]:
    if pid <= 1:
        return {
            "targetPid": pid,
            "targetType": "invalid",
            "targetId": pid,
            "terminated": False,
            "termSignalSent": False,
            "killSignalSent": False,
        }

    pgid = _resolve_terminatable_process_group(pid)
    target_type = "process-group" if pgid is not None else "pid"
    target_id = pgid if pgid is not None else pid

    term_signal_sent = _send_signal_to_target(pid, signal.SIGTERM, pgid)
    terminated = await _wait_for_target_exit(
        pid,
        pgid,
        timeout_sec=graceful_timeout_sec,
    )

    kill_signal_sent = False
    if not terminated:
        kill_signal_sent = _send_signal_to_target(pid, signal.SIGKILL, pgid)
        terminated = await _wait_for_target_exit(
            pid,
            pgid,
            timeout_sec=force_timeout_sec,
        )

    return {
        "targetPid": pid,
        "targetType": target_type,
        "targetId": target_id,
        "terminated": terminated,
        "termSignalSent": term_signal_sent,
        "killSignalSent": kill_signal_sent,
    }


def _resolve_external_shutdown_target_pid() -> Optional[int]:
    requested_pid = os.getenv("AURA_PIPELINE_CONTROLLER_PID", "").strip()
    if not requested_pid:
        return None

    try:
        parsed_pid = int(requested_pid)
    except ValueError:
        return None

    if parsed_pid <= 1:
        return None
    return parsed_pid


async def _wait_for_ice_gathering_complete(
    pc: RTCPeerConnection, timeout: float = 5.0
) -> None:
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_ice_gathering_state_change():
        if pc.iceGatheringState == "complete" and not done.is_set():
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        pass


def _gather_local_candidates(pc: RTCPeerConnection):
    candidates = []
    transports = set()

    if pc.sctp and pc.sctp.transport:
        transports.add(pc.sctp.transport)

    for transceiver in pc.getTransceivers():
        if transceiver.sender and transceiver.sender.transport:
            transports.add(transceiver.sender.transport)
        if transceiver.receiver and transceiver.receiver.transport:
            transports.add(transceiver.receiver.transport)

    for transport in transports:
        gatherer = getattr(transport, "iceGatherer", None)
        if gatherer:
            for cand in gatherer.getLocalCandidates() or []:
                cand_line = (
                    f"candidate:{cand.foundation} {cand.component} {cand.protocol} "
                    f"{cand.priority} {cand.ip} {cand.port} typ {cand.type}"
                )
                if getattr(cand, "tcpType", None):
                    cand_line += f" tcptype {cand.tcpType}"
                if getattr(cand, "relatedAddress", None):
                    cand_line += f" raddr {cand.relatedAddress}"
                if getattr(cand, "relatedPort", None):
                    cand_line += f" rport {cand.relatedPort}"

                candidates.append(
                    {
                        "candidate": cand_line,
                        "sdpMid": getattr(cand, "sdpMid", None),
                        "sdpMLineIndex": getattr(cand, "sdpMLineIndex", None),
                    }
                )
    return candidates


def _create_frame_source(cfg: StreamServerConfig):
    if cfg.input_source == "ros2":
        return Ros2ImageTopicFrameSource(
            image_topic=cfg.ros_image_topic,
            queue_size=cfg.ros_queue_size,
            slam_pose_topic=cfg.ros_slam_pose_topic,
            semantic_projected_map_topic=cfg.ros_semantic_projected_map_topic,
            semantic_octomap_cloud_topic=cfg.ros_semantic_octomap_cloud_topic,
        )
    if cfg.input_source == "video":
        return VideoFileFrameSource(cfg.video_path)
    raise ValueError(f"Unsupported input source: {cfg.input_source}")


app = create_app()
