from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from adapters.sensors.frame_source import AutoFrameSource, FrameSample, SyntheticFrameSource, build_synthetic_frame_sample
from adapters.sensors.isaac_live_source import IsaacLiveFrameSource, IsaacLiveSourceConfig
from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from systems.transport.bus.base import MessageBus
from systems.transport.bus.inproc_bus import InprocBus
from systems.transport.messages import FrameHeader
from systems.transport.shm import SharedMemoryRing
from systems.transport.bus.zmq_bus import ZmqBus


@dataclass(frozen=True)
class RuntimeIo:
    bus: MessageBus
    shm_ring: SharedMemoryRing | None

    def close(self, *, unlink_shm: bool = False) -> None:
        self.bus.close()
        if self.shm_ring is not None:
            self.shm_ring.close(unlink=unlink_shm)


def build_runtime_io(
    *,
    bus_kind: str,
    endpoint: str = "",
    bind: bool = False,
    shm_name: str,
    shm_slot_size: int,
    shm_capacity: int,
    create_shm: bool,
    role: str = "",
    control_endpoint: str = "",
    telemetry_endpoint: str = "",
    identity: str = "",
) -> RuntimeIo:
    kind = str(bus_kind).strip().lower()
    if kind == "inproc":
        return RuntimeIo(bus=InprocBus(), shm_ring=None)
    if kind != "zmq":
        raise ValueError(f"Unsupported bus kind: {bus_kind}")
    resolved_control = control_endpoint or endpoint
    if resolved_control == "":
        resolved_control = "tcp://127.0.0.1:5560"
    resolved_telemetry = telemetry_endpoint or derive_telemetry_endpoint(resolved_control)
    resolved_role = str(role).strip().lower() or ("bridge" if bool(bind) else "agent")
    shm_ring = SharedMemoryRing(
        name=str(shm_name),
        slot_size=int(shm_slot_size),
        capacity=int(shm_capacity),
        create=bool(create_shm),
    )
    return RuntimeIo(
        bus=ZmqBus(
            control_endpoint=str(resolved_control),
            telemetry_endpoint=str(resolved_telemetry),
            role=resolved_role,
            identity=str(identity),
        ),
        shm_ring=shm_ring,
    )


def derive_telemetry_endpoint(control_endpoint: str) -> str:
    endpoint = str(control_endpoint).strip()
    prefix, sep, port_str = endpoint.rpartition(":")
    if sep != "" and port_str.isdigit():
        return f"{prefix}:{int(port_str) + 1}"
    return f"{endpoint}.telemetry"


def build_frame_source(
    *,
    mode: str,
    scene: str,
    source_name: str,
    room_id: str = "",
    strict_live: bool = False,
    simulation_app=None,
    stage=None,
    env_provider=None,
    robot_pose_provider=None,
    robot_yaw_provider=None,
    sensor_factory=None,
    width: int = 96,
    height: int = 96,
):
    live_source = IsaacLiveFrameSource(
        simulation_app=simulation_app,
        stage=stage,
        env_provider=env_provider,
        robot_pose_provider=robot_pose_provider,
        robot_yaw_provider=robot_yaw_provider,
        sensor_factory=sensor_factory,
        config=IsaacLiveSourceConfig(
            source_name=f"{source_name}_live",
            strict_live=bool(strict_live),
            image_width=max(int(width), 96),
            image_height=max(int(height), 96),
        ),
    )
    fallback_source = SyntheticFrameSource(
        scene=scene,
        source_name=f"{source_name}_synthetic",
        room_id=room_id,
        width=width,
        height=height,
    )
    requested_mode = str(mode).strip().lower() or "auto"
    if requested_mode == "live":
        return live_source
    if requested_mode == "synthetic":
        return fallback_source
    return AutoFrameSource(live_source, fallback_source, strict_live=False)


def frame_sample_to_batch(sample: FrameSample) -> IsaacObservationBatch:
    return IsaacObservationBatch(
        frame_header=FrameHeader(
            frame_id=int(sample.frame_id),
            timestamp_ns=time.time_ns(),
            source=str(sample.source_name),
            width=int(sample.rgb.shape[1]),
            height=int(sample.rgb.shape[0]),
            camera_pose_xyz=tuple(float(v) for v in sample.camera_pose_xyz[:3]),
            camera_quat_wxyz=tuple(float(v) for v in sample.camera_quat_wxyz[:4]),
            robot_pose_xyz=tuple(float(v) for v in sample.robot_pose_xyz[:3]),
            robot_yaw_rad=float(sample.robot_yaw_rad),
            sim_time_s=float(sample.sim_time_s),
            metadata=dict(sample.metadata),
        ),
        robot_pose_xyz=tuple(float(v) for v in sample.robot_pose_xyz[:3]),
        robot_yaw_rad=float(sample.robot_yaw_rad),
        sim_time_s=float(sample.sim_time_s),
        rgb_image=np.asarray(sample.rgb, dtype=np.uint8),
        depth_image_m=np.asarray(sample.depth, dtype=np.float32),
        camera_intrinsic=np.asarray(sample.camera_intrinsic, dtype=np.float32),
        speaker_events=list(sample.speaker_events),
        capture_report=dict(sample.metadata.get("capture_report", {})) if isinstance(sample.metadata.get("capture_report"), dict) else {},
    )


def build_demo_batch(
    *,
    frame_id: int,
    scene: str,
    width: int = 96,
    height: int = 96,
    source: str,
    room_id: str = "",
) -> IsaacObservationBatch:
    sample = build_synthetic_frame_sample(
        frame_id=int(frame_id),
        scene=scene,
        source_name=source,
        room_id=room_id,
        width=width,
        height=height,
    )
    return frame_sample_to_batch(sample)


def infer_demo_scene(command_text: str, *, fallback: str = "apple") -> str:
    lowered = str(command_text).lower()
    if any(token in lowered for token in ("따라", "follow", "caller", "부르면", "사람", "person")):
        return "person"
    if any(token in lowered for token in ("cube", "큐브")):
        return "cube"
    if any(token in lowered for token in ("apple", "사과")):
        return "apple"
    return fallback
