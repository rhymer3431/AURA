from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from ipc.base import MessageBus
from ipc.inproc_bus import InprocBus
from ipc.messages import FrameHeader
from ipc.shm_ring import SharedMemoryRing
from ipc.zmq_bus import ZmqBus
from perception.speaker_events import SpeakerEvent


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
    endpoint: str,
    bind: bool,
    shm_name: str,
    shm_slot_size: int,
    shm_capacity: int,
    create_shm: bool,
) -> RuntimeIo:
    kind = str(bus_kind).strip().lower()
    if kind == "inproc":
        return RuntimeIo(bus=InprocBus(), shm_ring=None)
    if kind != "zmq":
        raise ValueError(f"Unsupported bus kind: {bus_kind}")
    shm_ring = SharedMemoryRing(
        name=str(shm_name),
        slot_size=int(shm_slot_size),
        capacity=int(shm_capacity),
        create=bool(create_shm),
    )
    return RuntimeIo(bus=ZmqBus(str(endpoint), bind=bool(bind)), shm_ring=shm_ring)


def build_demo_batch(
    *,
    frame_id: int,
    scene: str,
    width: int = 96,
    height: int = 96,
    source: str,
    room_id: str = "",
) -> IsaacObservationBatch:
    scene_name = str(scene).strip().lower()
    rgb = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    depth = np.full((int(height), int(width)), 1.5, dtype=np.float32)
    metadata: dict[str, object] = {"room_id": room_id}
    speaker_events: list[SpeakerEvent] = []

    if scene_name in {"apple", "cube"}:
        rgb[24:72, 28:68, 0] = 255
        metadata["target_class_hint"] = "apple" if scene_name == "apple" else "cube"
        metadata["color_hint"] = "red"
    elif scene_name == "person":
        rgb[18:78, 34:62, 2] = 255
        metadata["target_class_hint"] = "person"
        metadata["color_hint"] = "blue"
    else:
        rgb[24:72, 28:68, 1] = 255
        metadata["target_class_hint"] = scene_name or "object"
        metadata["color_hint"] = "green"

    if scene_name == "person":
        speaker_events.append(
            SpeakerEvent(
                timestamp=time.time(),
                direction_yaw_rad=0.45,
                speaker_id="person_0001",
                confidence=0.95,
                metadata={"source": "demo"},
            )
        )

    intrinsic = np.asarray(
        [
            [float(width), 0.0, float(width) * 0.5],
            [0.0, float(height), float(height) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return IsaacObservationBatch(
        frame_header=FrameHeader(
            frame_id=int(frame_id),
            timestamp_ns=time.time_ns(),
            source=str(source),
            width=int(width),
            height=int(height),
            camera_pose_xyz=(0.0, 0.0, 1.2),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            metadata=metadata,
        ),
        robot_pose_xyz=(0.0, 0.0, 0.0),
        rgb_image=rgb,
        depth_image_m=depth,
        camera_intrinsic=intrinsic,
        speaker_events=speaker_events,
    )


def infer_demo_scene(command_text: str, *, fallback: str = "apple") -> str:
    lowered = str(command_text).lower()
    if any(token in lowered for token in ("따라", "follow", "caller", "부르면", "사람", "person")):
        return "person"
    if any(token in lowered for token in ("cube", "큐브")):
        return "cube"
    if any(token in lowered for token in ("apple", "사과")):
        return "apple"
    return fallback
