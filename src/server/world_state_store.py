from __future__ import annotations

from dataclasses import replace
from typing import Any

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from ipc.messages import ActionCommand, ActionStatus
from runtime.planning_session import TrajectoryUpdate
from schemas.recovery import RecoveryState, RecoveryStateSnapshot
from schemas.execution_mode import normalize_execution_mode
from server.planner_runtime_state import PlannerRuntimeState

from schemas.commands import ResolvedCommand
from schemas.events import FrameEvent
from schemas.world_state import (
    ExecutionStateSnapshot,
    MemoryStateSnapshot,
    PerceptionStateSnapshot,
    PlanningStateSnapshot,
    RobotStateSnapshot,
    RuntimeStateSnapshot,
    SafetyStateSnapshot,
    TaskSnapshot,
    WorldStateSnapshot,
)


class WorldStateStore:
    def __init__(self, *, initial_mode: str = "", runtime: RuntimeStateSnapshot | None = None) -> None:
        normalized_mode = normalize_execution_mode(initial_mode)
        self._snapshot = WorldStateSnapshot(
            mode=normalized_mode,
            planning=PlanningStateSnapshot(planner_mode=normalized_mode),
            runtime=RuntimeStateSnapshot() if runtime is None else runtime,
        )

    def configure_runtime(self, *, runtime: RuntimeStateSnapshot) -> None:
        self._snapshot = replace(self._snapshot, runtime=runtime)

    def set_mode(self, mode: str) -> None:
        normalized = normalize_execution_mode(mode)
        self._snapshot = replace(
            self._snapshot,
            mode=normalized,
            planning=replace(self._snapshot.planning, planner_mode=normalized),
        )

    def update_task(self, task: TaskSnapshot) -> None:
        memory = replace(
            self._snapshot.memory,
            memory_aware_task_active=bool(task.mode == "MEM_NAV" and task.state == "active"),
        )
        self._snapshot = replace(self._snapshot, task=task, memory=memory)

    def seed_planning_state(self, *, mode: str, instruction: str, route_state: dict[str, object] | None = None) -> None:
        normalized = normalize_execution_mode(mode)
        self._snapshot = replace(
            self._snapshot,
            planning=replace(
                self._snapshot.planning,
                planner_mode=normalized,
                active_instruction=str(instruction),
                route_state=dict(route_state or {}),
            ),
        )

    def recovery_state(self) -> RecoveryStateSnapshot:
        return RecoveryStateSnapshot.from_dict(self._snapshot.safety.recovery_state.to_dict())

    def ingest_frame(self, frame_event: FrameEvent) -> None:
        sensor_health = dict(self._snapshot.robot.sensor_health)
        sensor_health.update(
            {
                "observation_available": bool(frame_event.observation is not None),
                "batch_available": bool(frame_event.batch is not None),
                "source": str(frame_event.source),
                "frame_id": int(frame_event.frame_id),
                "timestamp_ns": int(frame_event.timestamp_ns),
            }
        )
        stale_info = dict(self._snapshot.planning.stale_info)
        stale_info["last_frame_id"] = int(frame_event.frame_id)
        stale_info["last_frame_timestamp_ns"] = int(frame_event.timestamp_ns)
        self._snapshot = replace(
            self._snapshot,
            robot=replace(
                self._snapshot.robot,
                pose_xyz=tuple(float(v) for v in frame_event.robot_pose_xyz[:3]),
                yaw_rad=float(frame_event.robot_yaw_rad),
                frame_id=int(frame_event.frame_id),
                timestamp_ns=int(frame_event.timestamp_ns),
                source=str(frame_event.source),
                sensor_health=sensor_health,
                sensor_meta=dict(frame_event.sensor_meta),
            ),
            planning=replace(self._snapshot.planning, stale_info=stale_info),
            runtime=replace(self._snapshot.runtime, frame_available=bool(frame_event.observation is not None)),
        )

    def record_perception(
        self,
        batch: IsaacObservationBatch | None,
        *,
        summary: dict[str, object] | None = None,
    ) -> None:
        detector_summary = dict(summary or {})
        if batch is None:
            self._snapshot = replace(
                self._snapshot,
                perception=PerceptionStateSnapshot(
                    detector_backend=str(detector_summary.get("detector_backend", "")),
                    detector_selected_reason=str(detector_summary.get("detector_selected_reason", "")),
                    detector_ready=bool(detector_summary.get("detector_ready", False)),
                    detector_runtime_report=dict(detector_summary.get("detector_runtime_report", {}))
                    if isinstance(detector_summary.get("detector_runtime_report"), dict)
                    else {},
                ),
            )
            return
        overlay = batch.frame_header.metadata.get("viewer_overlay", {})
        detections = overlay.get("detections", []) if isinstance(overlay, dict) else []
        self._snapshot = replace(
            self._snapshot,
            perception=PerceptionStateSnapshot(
                summary={
                    "frame_id": int(batch.frame_header.frame_id),
                    "source": str(batch.frame_header.source),
                    "detection_count": len(batch.observations),
                    "speaker_event_count": len(batch.speaker_events),
                    "tracked_detection_count": len(detections) if isinstance(detections, list) else 0,
                    "capture_report": dict(batch.capture_report),
                },
                detector_backend=str(detector_summary.get("detector_backend", "")),
                detector_selected_reason=str(detector_summary.get("detector_selected_reason", "")),
                detector_ready=bool(detector_summary.get("detector_ready", False)),
                detector_runtime_report=dict(detector_summary.get("detector_runtime_report", {}))
                if isinstance(detector_summary.get("detector_runtime_report"), dict)
                else {},
                detection_count=len(batch.observations),
                tracked_detection_count=len(detections) if isinstance(detections, list) else 0,
                trajectory_point_count=len(overlay.get("trajectory_pixels", []))
                if isinstance(overlay.get("trajectory_pixels", []), list)
                else 0,
            ),
            robot=replace(self._snapshot.robot, capture_report=dict(batch.capture_report)),
        )

    def record_memory_context(
        self,
        memory_context,
        *,
        summary: dict[str, object] | None = None,
        task: TaskSnapshot | None = None,
    ) -> None:  # noqa: ANN001
        if memory_context is None and not summary:
            self._snapshot = replace(self._snapshot, memory=MemoryStateSnapshot())
            return
        memory_summary = dict(summary or {})
        scratchpad = memory_summary.get("scratchpad")
        memory_context_summary = {}
        if memory_context is not None:
            memory_context_summary = {
                "instruction": str(memory_context.instruction),
                "text_line_count": len(memory_context.text_lines),
                "keyframe_count": len(memory_context.keyframes),
                "crop_path": str(memory_context.crop_path),
                "latent_backend_hint": str(memory_context.latent_backend_hint),
            }
        memory_aware_task_active = False
        active_task = self._snapshot.task if task is None else task
        if active_task.mode == "MEM_NAV" and active_task.state == "active":
            memory_aware_task_active = True
        self._snapshot = replace(
            self._snapshot,
            memory=MemoryStateSnapshot(
                summary=memory_context_summary,
                object_count=int(memory_summary.get("object_count", 0) or 0),
                place_count=int(memory_summary.get("place_count", 0) or 0),
                semantic_rule_count=int(memory_summary.get("semantic_rule_count", 0) or 0),
                keyframe_count=int(memory_summary.get("keyframe_count", 0) or 0),
                scratchpad=dict(scratchpad) if isinstance(scratchpad, dict) else {},
                memory_aware_task_active=memory_aware_task_active,
            ),
        )

    def record_planning_result(
        self,
        update: TrajectoryUpdate,
        planner_state: PlannerRuntimeState | None = None,
        *,
        recovery_state: RecoveryStateSnapshot | None = None,
    ) -> None:
        stale_info = dict(self._snapshot.planning.stale_info)
        stale_info["planner_stale_sec"] = float(update.stale_sec)
        stale_info["goal_version"] = int(update.goal_version)
        stale_info["traj_version"] = int(update.traj_version)
        global_route: dict[str, object] = {}
        planner_control_reason = ""
        system2_pixel_goal = None
        if planner_state is not None:
            overlay = planner_state.viewer_overlay_state()
            planner_control_reason = str(overlay.get("planner_control_reason", ""))
            raw_pixel_goal = overlay.get("system2_pixel_goal")
            if isinstance(raw_pixel_goal, list) and len(raw_pixel_goal) >= 2:
                system2_pixel_goal = [int(raw_pixel_goal[0]), int(raw_pixel_goal[1])]
            global_route = {
                "enabled": bool(overlay.get("global_route_enabled", False)),
                "active": bool(overlay.get("global_route_active", False)),
                "waypoint_index": int(overlay.get("global_route_waypoint_index", 0) or 0),
                "waypoint_count": int(overlay.get("global_route_waypoint_count", 0) or 0),
                "last_replan_reason": str(overlay.get("global_route_last_replan_reason", "")),
                "last_error": str(overlay.get("global_route_last_error", "")),
                "goal_xy": list(overlay.get("global_route_goal_xy", []))
                if isinstance(overlay.get("global_route_goal_xy"), list)
                else [],
                "active_waypoint_xy": list(overlay.get("global_route_active_waypoint_xy", []))
                if isinstance(overlay.get("global_route_active_waypoint_xy"), list)
                else [],
                "waypoints_world": list(overlay.get("global_route_waypoints_world", []))
                if isinstance(overlay.get("global_route_waypoints_world"), list)
                else [],
            }
        active_instruction = str(self._snapshot.task.instruction)
        route_state = self._build_route_state(
            mode=self._snapshot.mode,
            update=update,
            global_route=global_route,
        )
        planning = PlanningStateSnapshot(
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
                "system2_pixel_goal": system2_pixel_goal,
            },
            active_nav_plan={
                "plan_version": int(update.plan_version),
                "goal_version": int(update.goal_version),
                "traj_version": int(update.traj_version),
                "trajectory_point_count": int(update.trajectory_world.shape[0]),
                "trajectory_world": [
                    [float(point[0]), float(point[1]), float(point[2]) if len(point) >= 3 else 0.0]
                    for point in np.asarray(update.trajectory_world, dtype=np.float32)
                ],
                "stop": bool(update.stop),
                "used_cached_traj": bool(update.used_cached_traj),
                "planner_control_reason": planner_control_reason,
                "global_route_waypoint_index": int(global_route.get("waypoint_index", 0) or 0),
                "global_route_waypoint_count": int(global_route.get("waypoint_count", 0) or 0),
            },
            plan_version=int(update.plan_version),
            goal_version=int(update.goal_version),
            traj_version=int(update.traj_version),
            planner_mode=self._snapshot.mode,
            active_instruction=active_instruction,
            route_state=route_state,
            planner_control_mode="" if update.planner_control_mode is None else str(update.planner_control_mode),
            planner_control_reason=planner_control_reason,
            planner_yaw_delta_rad=None
            if update.planner_yaw_delta_rad is None
            else float(update.planner_yaw_delta_rad),
            system2_pixel_goal=system2_pixel_goal,
            stale_info=stale_info,
            global_route=global_route,
        )
        self._snapshot = replace(
            self._snapshot,
            planning=planning,
            safety=self._safety_with_recovery(recovery_state),
        )

    def record_command_decision(self, resolved: ResolvedCommand, *, recovery_state: RecoveryStateSnapshot | None = None) -> None:
        action_command = resolved.action_command
        execution = ExecutionStateSnapshot(
            last_command_decision={
                "action_type": "" if action_command is None else str(action_command.action_type),
                "command_id": "" if action_command is None else str(action_command.command_id),
                "source": str(resolved.source),
                "safety_override": bool(resolved.safety_override),
                "status": None if resolved.status is None else str(resolved.status.state),
                "command_vector": [float(v) for v in resolved.command_vector.tolist()],
                "recovery_state": dict(resolved.metadata.get("recovery_state", {}))
                if isinstance(resolved.metadata.get("recovery_state"), dict)
                else {},
                "recovery_reason": str(resolved.metadata.get("recovery_reason", "")),
                "retry_count": int(resolved.metadata.get("retry_count", 0) or 0),
                "backoff_until_ns": int(resolved.metadata.get("backoff_until_ns", 0) or 0),
            },
            last_action_status=_status_summary(resolved.status),
            active_overrides={
                "safety_override": bool(resolved.safety_override),
                "recovery_state": dict(resolved.metadata.get("recovery_state", {}))
                if isinstance(resolved.metadata.get("recovery_state"), dict)
                else {},
            },
            locomotion_proposal_summary={
                "goal_distance_m": float(resolved.evaluation.goal_distance_m),
                "yaw_error_rad": float(resolved.evaluation.yaw_error_rad),
                "reached_goal": bool(resolved.evaluation.reached_goal),
                "force_stop": bool(resolved.evaluation.force_stop),
                "command_vector": [float(v) for v in resolved.command_vector.tolist()],
                "metadata": dict(resolved.metadata),
            },
            active_command_type="" if action_command is None else str(action_command.action_type),
            active_target=_active_target_summary(action_command),
        )
        self._snapshot = replace(
            self._snapshot,
            execution=execution,
            safety=self._safety_with_recovery(recovery_state),
        )

    def set_recovery_state(self, recovery_state: RecoveryStateSnapshot) -> None:
        self._snapshot = replace(
            self._snapshot,
            safety=self._safety_with_recovery(recovery_state),
        )

    def reset_recovery_state(self, *, entered_at_ns: int = 0, reason: str = "") -> None:
        self._snapshot = replace(
            self._snapshot,
            safety=self._safety_with_recovery(
                RecoveryStateSnapshot(
                    current_state=RecoveryState.NORMAL.value,
                    entered_at_ns=int(entered_at_ns),
                    retry_count=0,
                    backoff_until_ns=0,
                    last_trigger_reason=str(reason),
                )
            ),
        )

    def snapshot(self) -> WorldStateSnapshot:
        return WorldStateSnapshot.from_dict(self._snapshot.to_dict())

    def _safety_with_recovery(self, recovery_state: RecoveryStateSnapshot | None) -> SafetyStateSnapshot:
        current = self.recovery_state() if recovery_state is None else recovery_state
        state = current.state
        trigger = str(current.last_trigger_reason)
        return SafetyStateSnapshot(
            safe_stop=state in {RecoveryState.SAFE_STOP, RecoveryState.FAILED},
            stale=state in {RecoveryState.REPLAN_PENDING, RecoveryState.WAIT_SENSOR} or trigger == "trajectory_stale",
            timeout=trigger == "timeout",
            sensor_unavailable=state == RecoveryState.WAIT_SENSOR or trigger == "sensor_missing",
            recovery_state=current,
        )

    @staticmethod
    def _build_route_state(*, mode: str, update: TrajectoryUpdate, global_route: dict[str, object]) -> dict[str, object]:
        action_command = update.action_command
        action_metadata = {} if action_command is None else dict(action_command.metadata)
        normalized_mode = normalize_execution_mode(mode)
        if normalized_mode == "NAV":
            payload = {
                "pixelGoal": None,
                "plannerControlMode": "" if update.planner_control_mode is None else str(update.planner_control_mode),
                "plannerControlReason": str(action_metadata.get("planner_control_reason", "")),
            }
            raw_goal = action_metadata.get("system2_pixel_goal")
            if isinstance(raw_goal, list) and len(raw_goal) >= 2:
                payload["pixelGoal"] = [int(raw_goal[0]), int(raw_goal[1])]
            return payload
        if normalized_mode == "MEM_NAV":
            return {
                "memoryPoseXyz": list(action_metadata.get("memory_pose_xyz", []))
                if isinstance(action_metadata.get("memory_pose_xyz"), list)
                else [],
                "goalPoseXyz": list(action_metadata.get("goal_pose_xyz", []))
                if isinstance(action_metadata.get("goal_pose_xyz"), list)
                else [],
                "waypointIndex": int(global_route.get("waypoint_index", 0) or 0),
                "waypointCount": int(global_route.get("waypoint_count", 0) or 0),
            }
        if normalized_mode == "EXPLORE":
            return {"policy": "nogoal"}
        if normalized_mode == "TALK":
            return {"hookStatus": "ready"}
        return {}


def _status_summary(status: ActionStatus | None) -> dict[str, object]:
    if status is None:
        return {}
    summary: dict[str, object] = {
        "command_id": str(status.command_id),
        "state": str(status.state),
        "timestamp_ns": int(status.timestamp_ns),
        "success": bool(status.success),
        "reason": str(status.reason),
        "distance_remaining_m": status.distance_remaining_m,
        "metadata": dict(status.metadata),
    }
    if status.robot_pose_xyz is not None:
        summary["robot_pose_xyz"] = [float(value) for value in status.robot_pose_xyz[:3]]
    return summary


def _active_target_summary(command: ActionCommand | None) -> dict[str, object]:
    if command is None:
        return {}
    summary: dict[str, object] = {
        "action_type": str(command.action_type),
    }
    if command.target_track_id != "":
        summary["target_track_id"] = str(command.target_track_id)
    if command.target_place_id != "":
        summary["target_place_id"] = str(command.target_place_id)
    if command.target_object_id != "":
        summary["target_object_id"] = str(command.target_object_id)
    if command.target_person_id != "":
        summary["target_person_id"] = str(command.target_person_id)
    if command.target_pose_xyz is not None:
        summary["target_pose_xyz"] = [float(value) for value in command.target_pose_xyz[:3]]
    if command.look_at_yaw_rad is not None:
        summary["look_at_yaw_rad"] = float(command.look_at_yaw_rad)
    for key in (
        "target_mode",
        "target_class",
        "pose_source",
        "raw_target_pose_xyz",
        "filtered_target_pose_xyz",
        "nav_goal_pose_xyz",
        "approach_yaw_rad",
        "track_age_sec",
        "depth_m",
    ):
        if key in command.metadata:
            summary[key] = command.metadata[key]
    return summary
