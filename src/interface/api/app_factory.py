import asyncio
import json
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Set, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

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

from application.streaming.perception_loop import run_perception_stream
from application.streaming.runtime_manager import RuntimeManager
from infrastructure.streaming.config import StreamServerConfig
from infrastructure.streaming.websocket_sinks import WebSocketVideoSink, WebSocketMetadataSink
from infrastructure.streaming.webrtc_sinks import (
    PerceptionVideoTrack,
    WebRtcVideoSink,
    WebRtcMetadataSink,
)


def create_app() -> FastAPI:
    cfg = StreamServerConfig()
    runtime = RuntimeManager(cfg)
    pcs: Set[RTCPeerConnection] = set()

    app = FastAPI(title="Perception Streamer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
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

    @app.get("/")
    async def root():
        return JSONResponse(
            {
                "status": "ok",
                "message": "WebRTC offer at /webrtc/offer (metadata datachannel + video track). WebSocket legacy at /ws/stream",
                "videoPath": str(cfg.video_path),
                "targetFps": cfg.target_fps,
                "frameMaxWidth": cfg.frame_max_width,
                "jpegQuality": cfg.jpeg_quality,
                "llmModel": cfg.llm_model_name,
                "llmDevice": cfg.llm_device,
                "runtimeReady": runtime.perception is not None,
                "webrtcOfferPath": "/webrtc/offer",
                "webrtcDataChannel": "metadata",
            }
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

        try:
            await run_perception_stream(
                runtime.perception,
                runtime.scene_planner,
                video_sink,
                metadata_sink,
                cfg.video_path,
                cfg.target_fps,
                cfg.frame_max_width,
                stop_event,
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
                cfg.video_path,
                cfg.target_fps,
                cfg.frame_max_width,
                stop_event,
                on_shutdown,
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


app = create_app()
