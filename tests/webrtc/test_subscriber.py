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
from systems.transport.bus.inproc_bus import InprocBus
from systems.transport.messages import ActionStatus, CapabilityReport, FrameHeader, HealthPing, RuntimeNotice
from schemas.recovery import RecoveryStateSnapshot
from schemas.world_state import (
    ExecutionStateSnapshot,
    PerceptionStateSnapshot,
    PlanningStateSnapshot,
    RobotStateSnapshot,
    RuntimeStateSnapshot,
    SafetyStateSnapshot,
    TaskSnapshot,
    WorldStateSnapshot,
)
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
                            "planner_overlay": {"plan_version": 999, "interactive_instruction": "stale overlay"},
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
            bridge.publish_health(
                HealthPing(
                    component="aura_runtime",
                    details={
                        "worldState": WorldStateSnapshot(
                            task=TaskSnapshot(task_id="task-1", instruction="inspect apple", mode="interactive", state="active", command_id=8),
                            mode="interactive",
                            robot=RobotStateSnapshot(
                                pose_xyz=(1.0, 0.0, 0.0),
                                yaw_rad=0.5,
                                frame_id=3,
                                timestamp_ns=99,
                                source="unit_test",
                                sensor_health={"observation_available": True, "batch_available": True},
                            ),
                            perception=PerceptionStateSnapshot(
                                detector_backend="server-detector",
                                detection_count=5,
                                tracked_detection_count=5,
                                trajectory_point_count=2,
                            ),
                            planning=PlanningStateSnapshot(
                                plan_version=4,
                                goal_version=2,
                                traj_version=3,
                                planner_mode="interactive",
                                planner_control_mode="trajectory",
                                interactive_phase="task_active",
                                interactive_command_id=8,
                                interactive_instruction="inspect apple",
                                system2_pixel_goal=[24, 18],
                                stale_info={"planner_stale_sec": 0.4},
                            ),
                            execution=ExecutionStateSnapshot(
                                active_command_type="NAV_TO_POSE",
                                active_target={"action_type": "NAV_TO_POSE", "target_track_id": "server-track"},
                            ),
                            safety=SafetyStateSnapshot(
                                safe_stop=False,
                                stale=True,
                                timeout=False,
                                sensor_unavailable=False,
                                recovery_state=RecoveryStateSnapshot(
                                    current_state="REPLAN_PENDING",
                                    entered_at_ns=44,
                                    retry_count=1,
                                    backoff_until_ns=88,
                                    last_trigger_reason="trajectory_stale",
                                ),
                            ),
                            runtime=RuntimeStateSnapshot(frame_available=True),
                        ).to_dict()
                    },
                )
            )

            await asyncio.sleep(0.05)

            frame = subscriber.current_frame
            assert frame is not None
            assert frame.seq == 1
            assert frame.frame_header.frame_id == 3

            snapshot = subscriber.build_state_snapshot()
            assert snapshot["type"] == "snapshot"
            assert snapshot["frame_id"] == 3
            assert snapshot["planVersion"] == 4
            assert snapshot["active_command_type"] == "NAV_TO_POSE"
            assert snapshot["interactiveInstruction"] == "inspect apple"
            assert snapshot["recoveryState"] == "REPLAN_PENDING"
            assert snapshot["activeTarget"]["target_track_id"] == "server-track"

            telemetry = subscriber.build_frame_meta()
            assert telemetry is not None
            assert telemetry["detections"][0]["class_name"] == "apple"
            assert telemetry["trajectory_pixels"] == [[5, 6], [7, 8]]
            assert telemetry["planVersion"] == 4
            assert telemetry["stale"] is True
            assert telemetry["activeTarget"]["target_track_id"] == "server-track"

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
