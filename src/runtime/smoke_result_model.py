from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SmokeTierResult:
    tier: str
    status: str
    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SmokeResultSummary:
    overall_status: str
    sensor_status: SmokeTierResult
    pipeline_status: SmokeTierResult
    memory_status: SmokeTierResult
    frame_received: bool
    detection_attempted: bool
    detections_nonempty: bool
    memory_update_called: bool
    empty_observation_batch: bool
    recommended_next_action: str = ""
    target_tier: str = "sensor"

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "sensor_status": asdict(self.sensor_status),
            "pipeline_status": asdict(self.pipeline_status),
            "memory_status": asdict(self.memory_status),
            "frame_received": bool(self.frame_received),
            "detection_attempted": bool(self.detection_attempted),
            "detections_nonempty": bool(self.detections_nonempty),
            "memory_update_called": bool(self.memory_update_called),
            "empty_observation_batch": bool(self.empty_observation_batch),
            "recommended_next_action": self.recommended_next_action,
            "target_tier": self.target_tier,
        }


def aggregate_smoke_result(
    *,
    target_tier: str,
    frame_received: bool,
    rgb_ready: bool,
    depth_ready: bool,
    pose_ready: bool,
    detection_attempted: bool,
    detections_nonempty: bool,
    perception_ingress_ready: bool,
    memory_ingress_ready: bool,
    memory_update_called: bool,
) -> SmokeResultSummary:
    normalized_target = str(target_tier).strip().lower() or "sensor"
    sensor_pass = bool(frame_received and (rgb_ready or depth_ready) and pose_ready)
    pipeline_pass = bool(frame_received and detection_attempted and perception_ingress_ready)
    empty_observation_batch = bool(pipeline_pass and not detections_nonempty)
    memory_pass = bool(memory_ingress_ready and memory_update_called)

    sensor_status = SmokeTierResult(
        tier="sensor",
        status="sensor_smoke_pass" if sensor_pass else "sensor_smoke_failed",
        passed=sensor_pass,
        reason="" if sensor_pass else "live RGB/depth/pose ingress was incomplete",
        details={
            "frame_received": bool(frame_received),
            "rgb_ready": bool(rgb_ready),
            "depth_ready": bool(depth_ready),
            "pose_ready": bool(pose_ready),
        },
    )
    pipeline_status = SmokeTierResult(
        tier="pipeline",
        status="pipeline_smoke_pass" if pipeline_pass else "pipeline_smoke_failed",
        passed=pipeline_pass,
        reason="empty detection batch is allowed" if pipeline_pass and empty_observation_batch else ("" if pipeline_pass else "perception ingress did not complete"),
        details={
            "detection_attempted": bool(detection_attempted),
            "detections_nonempty": bool(detections_nonempty),
            "perception_ingress_ready": bool(perception_ingress_ready),
        },
    )
    if memory_pass:
        memory_status = SmokeTierResult(
            tier="memory",
            status="memory_smoke_pass",
            passed=True,
            details={
                "memory_ingress_ready": bool(memory_ingress_ready),
                "memory_update_called": bool(memory_update_called),
            },
        )
    elif empty_observation_batch:
        memory_status = SmokeTierResult(
            tier="memory",
            status="memory_empty_batch",
            passed=False,
            reason="Perception ingress succeeded, but no detections reached memory update.",
            details={
                "memory_ingress_ready": bool(memory_ingress_ready),
                "memory_update_called": bool(memory_update_called),
            },
        )
    else:
        memory_status = SmokeTierResult(
            tier="memory",
            status="memory_smoke_failed",
            passed=False,
            reason="memory update path did not run",
            details={
                "memory_ingress_ready": bool(memory_ingress_ready),
                "memory_update_called": bool(memory_update_called),
            },
        )

    if normalized_target == "sensor":
        overall = sensor_status.status if sensor_pass else "smoke_failed"
    elif normalized_target == "pipeline":
        overall = pipeline_status.status if pipeline_pass else ("sensor_smoke_pass" if sensor_pass else "smoke_failed")
    elif normalized_target == "memory":
        if memory_pass:
            overall = memory_status.status
        elif pipeline_pass:
            overall = pipeline_status.status
        elif sensor_pass:
            overall = sensor_status.status
        else:
            overall = "smoke_failed"
    else:
        overall = "full_smoke_pass" if sensor_pass and pipeline_pass and memory_pass else "smoke_partial"

    return SmokeResultSummary(
        overall_status=overall,
        sensor_status=sensor_status,
        pipeline_status=pipeline_status,
        memory_status=memory_status,
        frame_received=bool(frame_received),
        detection_attempted=bool(detection_attempted),
        detections_nonempty=bool(detections_nonempty),
        memory_update_called=bool(memory_update_called),
        empty_observation_batch=bool(empty_observation_batch),
        target_tier=normalized_target,
    )
