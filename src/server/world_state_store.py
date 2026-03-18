from __future__ import annotations

from dataclasses import replace
from typing import Any

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from runtime.planning_session import TrajectoryUpdate

from schemas.commands import ResolvedCommand
from schemas.events import FrameEvent
from schemas.world_state import TaskSnapshot, WorldStateSnapshot


class WorldStateStore:
    def __init__(self, *, initial_mode: str = "") -> None:
        self._snapshot = WorldStateSnapshot(mode=str(initial_mode))

    def set_mode(self, mode: str) -> None:
        self._snapshot = replace(self._snapshot, mode=str(mode))

    def update_task(self, task: TaskSnapshot) -> None:
        self._snapshot = replace(self._snapshot, current_task=task)

    def ingest_frame(self, frame_event: FrameEvent) -> None:
        stale = dict(self._snapshot.stale_timers)
        stale["last_frame_id"] = int(frame_event.frame_id)
        stale["last_frame_timestamp_ns"] = int(frame_event.timestamp_ns)
        self._snapshot = replace(
            self._snapshot,
            robot_pose_xyz=tuple(float(v) for v in frame_event.robot_pose_xyz[:3]),
            robot_yaw_rad=float(frame_event.robot_yaw_rad),
            stale_timers=stale,
        )

    def record_perception(self, batch: IsaacObservationBatch | None) -> None:
        if batch is None:
            self._snapshot = replace(self._snapshot, last_perception_summary={})
            return
        overlay = batch.frame_header.metadata.get("viewer_overlay", {})
        detections = overlay.get("detections", []) if isinstance(overlay, dict) else []
        self._snapshot = replace(
            self._snapshot,
            last_perception_summary={
                "frame_id": int(batch.frame_header.frame_id),
                "source": str(batch.frame_header.source),
                "detection_count": len(batch.observations),
                "speaker_event_count": len(batch.speaker_events),
                "tracked_detection_count": len(detections) if isinstance(detections, list) else 0,
                "capture_report": dict(batch.capture_report),
            },
        )

    def record_memory_context(self, memory_context) -> None:  # noqa: ANN001
        if memory_context is None:
            self._snapshot = replace(self._snapshot, last_memory_context={})
            return
        self._snapshot = replace(
            self._snapshot,
            last_memory_context={
                "instruction": str(memory_context.instruction),
                "text_line_count": len(memory_context.text_lines),
                "keyframe_count": len(memory_context.keyframes),
                "crop_path": str(memory_context.crop_path),
                "latent_backend_hint": str(memory_context.latent_backend_hint),
            },
        )

    def record_planning_result(self, update: TrajectoryUpdate) -> None:
        stale = dict(self._snapshot.stale_timers)
        stale["planner_stale_sec"] = float(update.stale_sec)
        self._snapshot = replace(
            self._snapshot,
            last_s2_result={
                "goal_version": int(update.goal_version),
                "traj_version": int(update.traj_version),
                "planner_control_mode": "" if update.planner_control_mode is None else str(update.planner_control_mode),
                "planner_yaw_delta_rad": None
                if update.planner_yaw_delta_rad is None
                else float(update.planner_yaw_delta_rad),
                "interactive_phase": "" if update.interactive_phase is None else str(update.interactive_phase),
                "interactive_command_id": int(update.interactive_command_id),
                "interactive_instruction": str(update.interactive_instruction),
            },
            active_nav_plan={
                "plan_version": int(update.plan_version),
                "goal_version": int(update.goal_version),
                "traj_version": int(update.traj_version),
                "trajectory_point_count": int(update.trajectory_world.shape[0]),
                "stop": bool(update.stop),
                "used_cached_traj": bool(update.used_cached_traj),
            },
            stale_timers=stale,
        )

    def record_command_decision(self, resolved: ResolvedCommand) -> None:
        self._snapshot = replace(
            self._snapshot,
            last_command_decision={
                "action_type": "" if resolved.action_command is None else str(resolved.action_command.action_type),
                "command_id": "" if resolved.action_command is None else str(resolved.action_command.command_id),
                "source": str(resolved.source),
                "safety_override": bool(resolved.safety_override),
                "status": None if resolved.status is None else str(resolved.status.state),
                "command_vector": [float(v) for v in resolved.command_vector.tolist()],
            },
        )

    def update_recovery_state(self, payload: dict[str, Any]) -> None:
        self._snapshot = replace(self._snapshot, recovery_state=dict(payload))

    def snapshot(self) -> WorldStateSnapshot:
        return replace(
            self._snapshot,
            current_task=self._snapshot.current_task,
            last_perception_summary=dict(self._snapshot.last_perception_summary),
            last_memory_context=dict(self._snapshot.last_memory_context),
            last_s2_result=dict(self._snapshot.last_s2_result),
            active_nav_plan=dict(self._snapshot.active_nav_plan),
            recovery_state=dict(self._snapshot.recovery_state),
            stale_timers=dict(self._snapshot.stale_timers),
            last_command_decision=dict(self._snapshot.last_command_decision),
        )
