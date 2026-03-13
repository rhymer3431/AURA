from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter, IsaacObservationBatch
from ipc.inproc_bus import InprocBus
from ipc.messages import ActionStatus, CapabilityReport, FrameHeader, HealthPing, RuntimeNotice
from webrtc.config import WebRTCGatewayConfig
from webrtc.subscriber import ObservationSubscriber


def test_observation_subscriber_caches_frames_and_events() -> None:
    async def scenario() -> None:
        bus = InprocBus()
        config = WebRTCGatewayConfig(stale_frame_timeout_sec=5.0)
        subscriber = ObservationSubscriber(config, bus=bus)
        listener = subscriber.add_listener()
        bridge = IsaacBridgeAdapter(bus)
        await subscriber.start()
        try:
            rgb = np.full((24, 24, 3), 127, dtype=np.uint8)
            depth = np.full((24, 24), 1.25, dtype=np.float32)
            bridge.publish_observation_batch(
                IsaacObservationBatch(
                    frame_header=FrameHeader(
                        frame_id=3,
                        timestamp_ns=99,
                        source="unit_test",
                        width=24,
                        height=24,
                        robot_pose_xyz=(1.0, 0.0, 0.0),
                        robot_yaw_rad=0.5,
                        sim_time_s=1.0,
                        metadata={
                            "viewer_overlay": {
                                "detector_backend": "stub",
                                "detections": [{"class_name": "apple", "bbox_xyxy": [1, 2, 10, 12], "track_id": "t1"}],
                                "trajectory_pixels": [[5, 6], [7, 8]],
                            }
                        },
                    ),
                    robot_pose_xyz=(1.0, 0.0, 0.0),
                    robot_yaw_rad=0.5,
                    sim_time_s=1.0,
                    rgb_image=rgb,
                    depth_image_m=depth,
                    camera_intrinsic=np.eye(3, dtype=np.float32),
                )
            )
            bridge.publish_status(ActionStatus(command_id="cmd-1", state="running"))
            bridge.publish_notice(RuntimeNotice(component="bridge", notice="ready"))
            bridge.publish_capability(CapabilityReport(component="sensor", status="ready"))
            bridge.publish_health(HealthPing(component="bridge"))

            await asyncio.sleep(0.05)

            frame = subscriber.current_frame
            assert frame is not None
            assert frame.seq == 1
            assert frame.frame_header.frame_id == 3

            snapshot = subscriber.build_state_snapshot()
            assert snapshot["type"] == "snapshot"
            assert snapshot["frame_id"] == 3

            telemetry = subscriber.build_frame_meta()
            assert telemetry is not None
            assert telemetry["detections"][0]["class_name"] == "apple"
            assert telemetry["trajectory_pixels"] == [[5, 6], [7, 8]]

            event_kinds = {
                (await asyncio.wait_for(listener.get(), timeout=1.0)).kind,
                (await asyncio.wait_for(listener.get(), timeout=1.0)).kind,
                (await asyncio.wait_for(listener.get(), timeout=1.0)).kind,
                (await asyncio.wait_for(listener.get(), timeout=1.0)).kind,
            }
            assert event_kinds == {"status", "notice", "capability", "health"}
        finally:
            await subscriber.close()

    asyncio.run(scenario())
