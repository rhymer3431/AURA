from __future__ import annotations

from dataclasses import dataclass, replace

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter, IsaacBridgeAdapterConfig, IsaacObservationBatch
from inference.detectors.factory import DetectorFactoryConfig
from ipc.base import MessageBus
from ipc.inproc_bus import InprocBus
from ipc.messages import ActionCommand, ActionStatus, CapabilityReport, HealthPing, RuntimeNotice, TaskRequest
from ipc.shm_ring import SharedMemoryRing
from perception.pipeline import PerceptionPipeline
from perception.viewer_overlay import build_viewer_overlay_payload
from services.memory_service import MemoryService
from services.task_orchestrator import TaskOrchestrator


@dataclass(frozen=True)
class SupervisorConfig:
    memory_db_path: str = "state/memory/memory.sqlite"
    detector_engine_path: str = ""
    detector_model_path: str = ""
    detector_device: str = ""


@dataclass(frozen=True)
class BusCycleResult:
    command: ActionCommand | None
    task_count: int
    frame_count: int
    status_count: int
    robot_pose: tuple[float, float, float]


class Supervisor:
    def __init__(
        self,
        *,
        bus: MessageBus | None = None,
        shm_ring: SharedMemoryRing | None = None,
        config: SupervisorConfig | None = None,
        memory_service: MemoryService | None = None,
        orchestrator: TaskOrchestrator | None = None,
        perception_pipeline: PerceptionPipeline | None = None,
    ) -> None:
        self.bus = bus or InprocBus()
        self.config = config or SupervisorConfig()
        self.memory_service = memory_service or MemoryService(db_path=self.config.memory_db_path)
        self.orchestrator = orchestrator or TaskOrchestrator(self.memory_service)
        self.perception_pipeline = perception_pipeline or PerceptionPipeline(
            detector_config=DetectorFactoryConfig(
                engine_path=self.config.detector_engine_path,
                model_path=self.config.detector_model_path,
                device=self.config.detector_device,
            )
        )
        self.bridge = IsaacBridgeAdapter(self.bus, IsaacBridgeAdapterConfig(), shm_ring=shm_ring)

    def submit_task(self, command_text: str, *, target_json: dict[str, object] | None = None, speaker_id: str = "") -> TaskRequest:
        request = TaskRequest(command_text=command_text, target_json=dict(target_json or {}), speaker_id=speaker_id)
        self.bridge.publish_task_request(request)
        self.orchestrator.submit_task(request)
        return request

    def submit_task_request(self, request: TaskRequest) -> TaskRequest:
        self.bridge.publish_task_request(request)
        self.orchestrator.submit_task(request)
        return request

    def ingest(self, batch: IsaacObservationBatch, *, publish: bool = True) -> None:
        if publish:
            self.bridge.publish_observation_batch(batch)
        if batch.speaker_events:
            for event in batch.speaker_events:
                self.orchestrator.on_speaker_event(event)
        if batch.observations:
            self.orchestrator.on_observations(batch.observations)

    def process_frame(self, batch: IsaacObservationBatch, *, publish: bool = True) -> IsaacObservationBatch:
        if batch.rgb_image is None or batch.depth_image_m is None:
            self.ingest(batch, publish=publish)
            return batch
        intrinsic = batch.camera_intrinsic
        if intrinsic is None:
            intrinsic = self._default_intrinsic(batch.frame_header.width, batch.frame_header.height)
        frame_result = self.perception_pipeline.process_frame(
            rgb_image=batch.rgb_image,
            depth_image_m=batch.depth_image_m,
            timestamp=batch.frame_header.timestamp_ns / 1.0e9,
            camera_pose_xyz=batch.frame_header.camera_pose_xyz,
            camera_quat_wxyz=batch.frame_header.camera_quat_wxyz,
            camera_intrinsic=intrinsic,
            metadata={
                **dict(batch.frame_header.metadata),
                "robot_pose_xyz": list(batch.robot_pose_xyz),
                "robot_yaw_rad": float(batch.robot_yaw_rad),
                "sim_time_s": float(batch.sim_time_s),
                "frame_source": str(batch.frame_header.source),
                "capture_report": dict(batch.capture_report),
            },
        )
        enriched_header = replace(
            batch.frame_header,
            metadata={
                **dict(batch.frame_header.metadata),
                "viewer_overlay": build_viewer_overlay_payload(frame_result),
            },
        )
        enriched = IsaacObservationBatch(
            frame_header=enriched_header,
            robot_pose_xyz=batch.robot_pose_xyz,
            robot_yaw_rad=batch.robot_yaw_rad,
            sim_time_s=batch.sim_time_s,
            rgb_image=batch.rgb_image,
            depth_image_m=batch.depth_image_m,
            camera_intrinsic=intrinsic,
            observations=frame_result.observations,
            speaker_events=[*batch.speaker_events, *frame_result.speaker_events],
            capture_report=dict(batch.capture_report),
        )
        self.ingest(enriched, publish=publish)
        return enriched

    def step(
        self,
        *,
        now: float,
        robot_pose: tuple[float, float, float],
        action_status: ActionStatus | None = None,
        publish: bool = True,
        publish_status: bool = False,
    ) -> ActionCommand | None:
        if action_status is not None and publish and publish_status:
            self.bridge.publish_status(action_status)
        command = self.orchestrator.step(now=now, robot_pose=robot_pose, action_status=action_status)
        if command is not None and publish:
            self.bus.publish(self.bridge.config.command_topic, command)
        return command

    def run_bus_cycle(
        self,
        *,
        now: float,
        robot_pose: tuple[float, float, float] | None = None,
    ) -> ActionCommand | None:
        return self.run_bus_cycle_result(now=now, robot_pose=robot_pose).command

    def run_bus_cycle_result(
        self,
        *,
        now: float,
        robot_pose: tuple[float, float, float] | None = None,
    ) -> BusCycleResult:
        latest_robot_pose = robot_pose
        task_requests = self.bridge.drain_task_requests()
        for request in task_requests:
            self.orchestrator.submit_task(request)
        frame_headers = self.bridge.drain_frame_headers()
        for frame_header in frame_headers:
            batch = self.bridge.reconstruct_batch(frame_header)
            latest_robot_pose = batch.robot_pose_xyz
            self.process_frame(batch, publish=False)
        statuses = self.bridge.drain_statuses()
        latest_status = statuses[-1] if statuses else None
        pose = latest_robot_pose or (0.0, 0.0, 0.0)
        command = self.step(now=now, robot_pose=pose, action_status=latest_status, publish=True)
        return BusCycleResult(
            command=command,
            task_count=len(task_requests),
            frame_count=len(frame_headers),
            status_count=len(statuses),
            robot_pose=tuple(float(v) for v in pose[:3]),
        )

    def snapshot(self) -> dict[str, object]:
        snapshot = self.orchestrator.snapshot()
        snapshot["detector_backend"] = self.perception_pipeline.detector.info.backend_name
        snapshot["detector_selected_reason"] = self.perception_pipeline.detector.info.selected_reason
        snapshot["detector_runtime_report"] = (
            None
            if self.perception_pipeline.detector.runtime_report is None
            else self.perception_pipeline.detector.runtime_report.as_dict()
        )
        return snapshot

    def publish_runtime_diagnostics(self) -> None:
        detector_report = self.perception_pipeline.detector.runtime_report
        if detector_report is not None:
            self.bridge.publish_capability(
                CapabilityReport(
                    component="detector",
                    status="ready" if detector_report.ready_for_inference else "fallback",
                    backend_name=self.perception_pipeline.detector.info.backend_name,
                    details=detector_report.as_dict(),
                    warnings=list(detector_report.warnings),
                    errors=list(detector_report.errors),
                )
            )
        self.bridge.publish_health(
            HealthPing(
                component="memory_agent" if not isinstance(self.bus, InprocBus) else "local_stack",
                details={"snapshot": self.snapshot()},
            )
        )

    def publish_notice(self, level: str, notice: str, *, details: dict[str, object] | None = None) -> None:
        self.bridge.publish_notice(
            RuntimeNotice(component="supervisor", level=level, notice=notice, details=dict(details or {}))
        )

    @staticmethod
    def _default_intrinsic(width: int, height: int):
        w = max(int(width), 1)
        h = max(int(height), 1)
        focal = float(max(w, h))
        return __import__("numpy").asarray(
            [[focal, 0.0, w * 0.5], [0.0, focal, h * 0.5], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
