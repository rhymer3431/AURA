from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter, IsaacObservationBatch
from apps.webrtc_gateway_app import create_app, parse_args
from ipc.inproc_bus import InprocBus
from ipc.messages import FrameHeader
from webrtc.config import WebRTCGatewayConfig
from webrtc.subscriber import ObservationSubscriber


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_parse_args_exposes_webrtc_defaults() -> None:
    args = parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8090
    assert args.control_endpoint == "tcp://127.0.0.1:5580"
    assert args.telemetry_endpoint == "tcp://127.0.0.1:5581"
    assert args.enable_depth_track is True
    assert args.rgb_fps == 15.0
    assert args.depth_fps == 5.0


def test_webrtc_gateway_offer_loopback() -> None:
    pytest.importorskip("aiohttp")
    pytest.importorskip("aiortc")

    async def wait_for_ice_complete(peer_connection) -> None:  # noqa: ANN001
        if peer_connection.iceGatheringState == "complete":
            return
        done = asyncio.Event()

        @peer_connection.on("icegatheringstatechange")
        def _on_change() -> None:
            if peer_connection.iceGatheringState == "complete":
                done.set()

        await asyncio.wait_for(done.wait(), timeout=10.0)

    async def scenario() -> None:
        from aiohttp import ClientSession, web
        from aiortc import RTCPeerConnection, RTCSessionDescription

        bus = InprocBus()
        config = WebRTCGatewayConfig(
            control_endpoint="inproc://unused",
            telemetry_endpoint="inproc://unused",
            stale_frame_timeout_sec=30.0,
        )
        subscriber = ObservationSubscriber(config, bus=bus)
        args = parse_args([])
        app = create_app(args, subscriber=subscriber)
        runner = web.AppRunner(app)
        await runner.setup()
        port = _free_tcp_port()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()

        bridge = IsaacBridgeAdapter(bus)
        bridge.publish_observation_batch(
            IsaacObservationBatch(
                frame_header=FrameHeader(
                    frame_id=4,
                    timestamp_ns=123,
                    source="loopback",
                    width=32,
                    height=32,
                    robot_pose_xyz=(0.0, 0.0, 0.0),
                    robot_yaw_rad=0.0,
                    sim_time_s=2.0,
                    metadata={
                        "viewer_overlay": {
                            "detector_backend": "stub",
                            "detections": [{"class_name": "apple", "bbox_xyxy": [1, 2, 10, 12], "track_id": "t1"}],
                            "trajectory_pixels": [[4, 5], [6, 7]],
                        }
                    },
                ),
                robot_pose_xyz=(0.0, 0.0, 0.0),
                robot_yaw_rad=0.0,
                sim_time_s=2.0,
                rgb_image=np.full((32, 32, 3), 255, dtype=np.uint8),
                depth_image_m=np.full((32, 32), 1.0, dtype=np.float32),
                camera_intrinsic=np.eye(3, dtype=np.float32),
            )
        )
        await asyncio.sleep(0.05)

        pc = RTCPeerConnection()
        state_messages: list[dict[str, object]] = []
        telemetry_messages: list[dict[str, object]] = []

        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("video", direction="recvonly")
        state_channel = pc.createDataChannel("state")
        telemetry_channel = pc.createDataChannel("telemetry", ordered=False, maxRetransmits=0)

        @state_channel.on("message")
        def _on_state_message(message) -> None:  # noqa: ANN001
            state_messages.append(json.loads(str(message)))

        @telemetry_channel.on("message")
        def _on_telemetry_message(message) -> None:  # noqa: ANN001
            telemetry_messages.append(json.loads(str(message)))

        try:
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await wait_for_ice_complete(pc)
            async with ClientSession() as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}/offer",
                    json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
                )
                assert response.status == 200
                body = await response.json()
            await pc.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type=body["type"]))

            deadline = time.time() + 10.0
            while time.time() < deadline:
                has_state = any(item.get("type") == "session_ready" for item in state_messages)
                has_meta = any(item.get("type") == "frame_meta" for item in telemetry_messages)
                if has_state and has_meta:
                    break
                await asyncio.sleep(0.05)

            assert any(item.get("type") == "session_ready" for item in state_messages)
            assert any(item.get("type") == "snapshot" for item in state_messages)
            frame_meta = next(item for item in telemetry_messages if item.get("type") == "frame_meta")
            assert frame_meta["frame_id"] == 4
            assert frame_meta["detections"][0]["class_name"] == "apple"
        finally:
            await pc.close()
            await runner.cleanup()

    asyncio.run(scenario())
