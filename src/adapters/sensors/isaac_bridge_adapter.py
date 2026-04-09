from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from systems.transport.bus.base import MessageBus
from systems.transport.frame_codec import decode_ndarray, encode_ndarray, ref_from_dict, ref_to_dict
from systems.transport.messages import (
    ActionCommand,
    ActionStatus,
    CapabilityReport,
    FrameHeader,
    HealthPing,
    RuntimeControlRequest,
    RuntimeNotice,
    TaskRequest,
)
from systems.transport.shm import SharedMemoryRing
from memory.models import ObsObject
from perception.speaker_events import SpeakerEvent


@dataclass(frozen=True)
class IsaacBridgeAdapterConfig:
    observation_topic: str = "isaac.observation"
    command_topic: str = "isaac.command"
    status_topic: str = "isaac.status"
    task_topic: str = "isaac.task"
    runtime_control_topic: str = "isaac.runtime_control"
    notice_topic: str = "isaac.notice"
    capability_topic: str = "isaac.capability"
    health_topic: str = "isaac.health"


@dataclass
class IsaacObservationBatch:
    frame_header: FrameHeader
    robot_pose_xyz: tuple[float, float, float]
    robot_yaw_rad: float = 0.0
    sim_time_s: float = 0.0
    rgb_image: np.ndarray | None = None
    depth_image_m: np.ndarray | None = None
    camera_intrinsic: np.ndarray | None = None
    observations: list[ObsObject] = field(default_factory=list)
    speaker_events: list[SpeakerEvent] = field(default_factory=list)
    capture_report: dict[str, object] = field(default_factory=dict)


class IsaacBridgeAdapter:
    def __init__(self, bus: MessageBus, config: IsaacBridgeAdapterConfig | None = None, *, shm_ring: SharedMemoryRing | None = None) -> None:
        self._bus = bus
        self.config = config or IsaacBridgeAdapterConfig()
        self._shm_ring = shm_ring
        self._latest_batch: IsaacObservationBatch | None = None

    @property
    def latest_batch(self) -> IsaacObservationBatch | None:
        return self._latest_batch

    def publish_observation_batch(self, batch: IsaacObservationBatch) -> None:
        frame_header = batch.frame_header
        if self._shm_ring is not None and batch.rgb_image is not None:
            rgb_ref = self._shm_ring.write(encode_ndarray(batch.rgb_image))
            frame_header = FrameHeader(
                frame_id=batch.frame_header.frame_id,
                timestamp_ns=batch.frame_header.timestamp_ns,
                source=batch.frame_header.source,
                rgb_shm=rgb_ref.name,
                depth_shm=batch.frame_header.depth_shm,
                width=batch.frame_header.width,
                height=batch.frame_header.height,
                rgb_encoding=batch.frame_header.rgb_encoding,
                depth_encoding=batch.frame_header.depth_encoding,
                camera_pose_xyz=batch.frame_header.camera_pose_xyz,
                camera_quat_wxyz=batch.frame_header.camera_quat_wxyz,
                robot_pose_xyz=batch.frame_header.robot_pose_xyz,
                robot_yaw_rad=float(batch.frame_header.robot_yaw_rad),
                sim_time_s=float(batch.frame_header.sim_time_s),
                metadata={**batch.frame_header.metadata, "rgb_ref": ref_to_dict(rgb_ref)},
                message_id=batch.frame_header.message_id,
            )
        elif batch.rgb_image is not None:
            frame_header = FrameHeader(
                frame_id=batch.frame_header.frame_id,
                timestamp_ns=batch.frame_header.timestamp_ns,
                source=batch.frame_header.source,
                rgb_shm=batch.frame_header.rgb_shm,
                depth_shm=batch.frame_header.depth_shm,
                width=batch.frame_header.width,
                height=batch.frame_header.height,
                rgb_encoding=batch.frame_header.rgb_encoding,
                depth_encoding=batch.frame_header.depth_encoding,
                camera_pose_xyz=batch.frame_header.camera_pose_xyz,
                camera_quat_wxyz=batch.frame_header.camera_quat_wxyz,
                robot_pose_xyz=batch.frame_header.robot_pose_xyz,
                robot_yaw_rad=float(batch.frame_header.robot_yaw_rad),
                sim_time_s=float(batch.frame_header.sim_time_s),
                metadata={**batch.frame_header.metadata, "rgb_inline": encode_ndarray(batch.rgb_image).hex()},
                message_id=batch.frame_header.message_id,
            )
        if self._shm_ring is not None and batch.depth_image_m is not None:
            depth_ref = self._shm_ring.write(encode_ndarray(batch.depth_image_m))
            frame_header = FrameHeader(
                frame_id=frame_header.frame_id,
                timestamp_ns=frame_header.timestamp_ns,
                source=frame_header.source,
                rgb_shm=frame_header.rgb_shm,
                depth_shm=depth_ref.name,
                width=frame_header.width,
                height=frame_header.height,
                rgb_encoding=frame_header.rgb_encoding,
                depth_encoding=frame_header.depth_encoding,
                camera_pose_xyz=frame_header.camera_pose_xyz,
                camera_quat_wxyz=frame_header.camera_quat_wxyz,
                robot_pose_xyz=batch.robot_pose_xyz,
                robot_yaw_rad=float(batch.robot_yaw_rad),
                sim_time_s=float(batch.sim_time_s),
                metadata={
                    **frame_header.metadata,
                    "depth_ref": ref_to_dict(depth_ref),
                    "camera_intrinsic": batch.camera_intrinsic.tolist() if batch.camera_intrinsic is not None else None,
                    "capture_report": dict(batch.capture_report),
                    "speaker_events": [
                        {
                            "timestamp": event.timestamp,
                            "direction_yaw_rad": event.direction_yaw_rad,
                            "speaker_id": event.speaker_id,
                            "confidence": event.confidence,
                            **event.metadata,
                        }
                        for event in batch.speaker_events
                    ],
                },
                message_id=frame_header.message_id,
            )
        elif batch.depth_image_m is not None:
            frame_header = FrameHeader(
                frame_id=frame_header.frame_id,
                timestamp_ns=frame_header.timestamp_ns,
                source=frame_header.source,
                rgb_shm=frame_header.rgb_shm,
                depth_shm=frame_header.depth_shm,
                width=frame_header.width,
                height=frame_header.height,
                rgb_encoding=frame_header.rgb_encoding,
                depth_encoding=frame_header.depth_encoding,
                camera_pose_xyz=frame_header.camera_pose_xyz,
                camera_quat_wxyz=frame_header.camera_quat_wxyz,
                robot_pose_xyz=batch.robot_pose_xyz,
                robot_yaw_rad=float(batch.robot_yaw_rad),
                sim_time_s=float(batch.sim_time_s),
                metadata={
                    **frame_header.metadata,
                    "depth_inline": encode_ndarray(batch.depth_image_m).hex(),
                    "camera_intrinsic": batch.camera_intrinsic.tolist() if batch.camera_intrinsic is not None else None,
                    "capture_report": dict(batch.capture_report),
                    "speaker_events": [
                        {
                            "timestamp": event.timestamp,
                            "direction_yaw_rad": event.direction_yaw_rad,
                            "speaker_id": event.speaker_id,
                            "confidence": event.confidence,
                            **event.metadata,
                        }
                        for event in batch.speaker_events
                    ],
                },
                message_id=frame_header.message_id,
            )
        self._latest_batch = batch
        self._bus.publish(self.config.observation_topic, frame_header)

    def publish_task_request(self, request: TaskRequest) -> None:
        self._bus.publish(self.config.task_topic, request)

    def publish_runtime_control(self, request: RuntimeControlRequest) -> None:
        self._bus.publish(self.config.runtime_control_topic, request)

    def publish_status(self, status: ActionStatus) -> None:
        self._bus.publish(self.config.status_topic, status)

    def publish_notice(self, notice: RuntimeNotice) -> None:
        self._bus.publish(self.config.notice_topic, notice)

    def publish_capability(self, report: CapabilityReport) -> None:
        self._bus.publish(self.config.capability_topic, report)

    def publish_health(self, ping: HealthPing) -> None:
        self._bus.publish(self.config.health_topic, ping)

    def drain_commands(self) -> list[ActionCommand]:
        return [record.message for record in self._bus.poll(self.config.command_topic, max_items=32)]

    def drain_task_requests(self) -> list[TaskRequest]:
        return [record.message for record in self._bus.poll(self.config.task_topic, max_items=32)]

    def drain_runtime_controls(self) -> list[RuntimeControlRequest]:
        return [record.message for record in self._bus.poll(self.config.runtime_control_topic, max_items=32)]

    def drain_statuses(self) -> list[ActionStatus]:
        return [record.message for record in self._bus.poll(self.config.status_topic, max_items=32)]

    def drain_notices(self) -> list[RuntimeNotice]:
        return [record.message for record in self._bus.poll(self.config.notice_topic, max_items=32)]

    def drain_capabilities(self) -> list[CapabilityReport]:
        return [record.message for record in self._bus.poll(self.config.capability_topic, max_items=32)]

    def drain_health(self) -> list[HealthPing]:
        return [record.message for record in self._bus.poll(self.config.health_topic, max_items=32)]

    def drain_frame_headers(self) -> list[FrameHeader]:
        return [record.message for record in self._bus.poll(self.config.observation_topic, max_items=32)]

    def reconstruct_batch(self, frame_header: FrameHeader) -> IsaacObservationBatch:
        rgb = None
        depth = None
        intrinsic = None
        metadata = dict(frame_header.metadata)
        if self._shm_ring is not None and isinstance(metadata.get("rgb_ref"), dict):
            rgb = decode_ndarray(self._shm_ring.read(ref_from_dict(metadata["rgb_ref"])))
        elif isinstance(metadata.get("rgb_inline"), str) and metadata.get("rgb_inline") != "":
            rgb = decode_ndarray(bytes.fromhex(str(metadata["rgb_inline"])))
        if self._shm_ring is not None and isinstance(metadata.get("depth_ref"), dict):
            depth = decode_ndarray(self._shm_ring.read(ref_from_dict(metadata["depth_ref"])))
        elif isinstance(metadata.get("depth_inline"), str) and metadata.get("depth_inline") != "":
            depth = decode_ndarray(bytes.fromhex(str(metadata["depth_inline"])))
        if isinstance(metadata.get("camera_intrinsic"), list):
            intrinsic = np.asarray(metadata.get("camera_intrinsic"), dtype=np.float32)
        speaker_events: list[SpeakerEvent] = []
        for event in metadata.get("speaker_events", []):
            if not isinstance(event, dict):
                continue
            speaker_events.append(
                SpeakerEvent(
                    timestamp=float(event.get("timestamp", 0.0)),
                    direction_yaw_rad=float(event.get("direction_yaw_rad", 0.0)),
                    speaker_id=str(event.get("speaker_id", "")),
                    confidence=float(event.get("confidence", 1.0)),
                    metadata={key: value for key, value in event.items() if key not in {"timestamp", "direction_yaw_rad", "speaker_id", "confidence"}},
                )
            )
        return IsaacObservationBatch(
            frame_header=frame_header,
            robot_pose_xyz=tuple(float(v) for v in frame_header.robot_pose_xyz[:3]),
            robot_yaw_rad=float(frame_header.robot_yaw_rad),
            sim_time_s=float(frame_header.sim_time_s),
            rgb_image=None if rgb is None else np.asarray(rgb, dtype=np.uint8),
            depth_image_m=None if depth is None else np.asarray(depth, dtype=np.float32),
            camera_intrinsic=intrinsic,
            speaker_events=speaker_events,
            capture_report=dict(metadata.get("capture_report", {})),
        )
