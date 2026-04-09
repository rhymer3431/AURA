from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from common.geometry import quat_wxyz_to_yaw
from systems.transport.messages import FrameHeader
from schemas.events import FrameEvent, WorkerMetadata


@dataclass(frozen=True)
class FrameCaptureResult:
    base_state: object
    robot_pose_xyz: tuple[float, float, float]
    robot_quat_wxyz: np.ndarray
    robot_yaw_rad: float
    runtime_events: tuple[object, ...]
    observation: object | None
    batch: IsaacObservationBatch | None
    frame_event: FrameEvent


@dataclass(slots=True)
class FrameIngestor:
    controller_provider: Callable[[], object | None]
    planning_session_provider: Callable[[], object]
    supervisor_provider: Callable[[], object]
    runtime_events_provider: Callable[[], list[object]]
    planner_overlay_provider: Callable[[], dict[str, object]]
    viewer_publish: bool
    timeout_sec: float

    def capture(self, frame_idx: int) -> FrameCaptureResult:
        controller = self.controller_provider()
        if controller is None:
            raise RuntimeError("AuraRuntimeCommandSource.initialize() must be called before update().")

        base_state = controller.get_base_state()
        robot_pose = tuple(float(v) for v in np.asarray(base_state.position_w, dtype=np.float32).reshape(-1)[:3])
        robot_quat = np.asarray(base_state.quat_wxyz, dtype=np.float32)
        robot_yaw = float(quat_wxyz_to_yaw(robot_quat))
        runtime_events = tuple(self.runtime_events_provider())

        planning_session = self.planning_session_provider()
        observation = planning_session.capture_observation(frame_idx)
        batch = None
        if observation is not None:
            timestamp_ns = time.time_ns()
            sim_time_s = float(time.time())
            batch = IsaacObservationBatch(
                frame_header=FrameHeader(
                    frame_id=int(observation.frame_id),
                    timestamp_ns=timestamp_ns,
                    source="aura_runtime",
                    width=int(observation.rgb.shape[1]),
                    height=int(observation.rgb.shape[0]),
                    camera_pose_xyz=tuple(float(v) for v in observation.cam_pos[:3]),
                    camera_quat_wxyz=tuple(float(v) for v in observation.cam_quat[:4]),
                    robot_pose_xyz=robot_pose,
                    robot_yaw_rad=float(robot_yaw),
                    sim_time_s=sim_time_s,
                    metadata={
                        **dict(observation.sensor_meta),
                        "planner_overlay": dict(self.planner_overlay_provider()),
                    },
                ),
                robot_pose_xyz=robot_pose,
                robot_yaw_rad=float(robot_yaw),
                sim_time_s=sim_time_s,
                rgb_image=observation.rgb,
                depth_image_m=observation.depth,
                camera_intrinsic=observation.intrinsic,
                capture_report=dict(observation.sensor_meta),
            )
        supervisor = self.supervisor_provider()
        frame_timestamp_ns = time.time_ns()
        sim_time_s = float(time.time())
        frame_event = FrameEvent(
            metadata=WorkerMetadata(
                task_id=supervisor.memory_service.scratchpad.task_id,
                frame_id=int(frame_idx),
                timestamp_ns=frame_timestamp_ns,
                source="aura_runtime",
                timeout_ms=int(max(float(self.timeout_sec) * 1000.0, 0.0)),
            ),
            frame_id=int(frame_idx),
            timestamp_ns=frame_timestamp_ns,
            source="aura_runtime",
            robot_pose_xyz=robot_pose,
            robot_yaw_rad=float(robot_yaw),
            sim_time_s=sim_time_s,
            observation=observation,
            batch=batch,
            sensor_meta={} if observation is None else dict(observation.sensor_meta),
            planner_overlay={}
            if batch is None
            else dict(batch.frame_header.metadata.get("planner_overlay", {})),
            publish_observation=bool(self.viewer_publish),
        )
        return FrameCaptureResult(
            base_state=base_state,
            robot_pose_xyz=robot_pose,
            robot_quat_wxyz=robot_quat,
            robot_yaw_rad=float(robot_yaw),
            runtime_events=runtime_events,
            observation=observation,
            batch=batch,
            frame_event=frame_event,
        )
