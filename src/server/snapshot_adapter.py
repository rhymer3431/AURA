from __future__ import annotations

import math
from typing import Any

from schemas.world_state import WorldStateSnapshot


class SnapshotAdapter:
    @staticmethod
    def _optional_float(value: object) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _int_pair(value: object) -> list[int] | None:
        if isinstance(value, list) and len(value) >= 2:
            return [int(value[0]), int(value[1])]
        return None

    @staticmethod
    def _float_triplet(value: object) -> list[float] | None:
        if isinstance(value, list) and len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
        return None

    @staticmethod
    def _bbox_quad(value: object) -> list[int] | None:
        if isinstance(value, list) and len(value) == 4:
            return [int(value[0]), int(value[1]), int(value[2]), int(value[3])]
        return None

    @staticmethod
    def _nav_trajectory_world(world_state: WorldStateSnapshot) -> list[list[float]]:
        raw_points = world_state.planning.active_nav_plan.get("trajectory_world", [])
        compact: list[list[float]] = []
        if isinstance(raw_points, list):
            for point in raw_points:
                if isinstance(point, list) and len(point) >= 2:
                    x = float(point[0])
                    y = float(point[1])
                    z = float(point[2]) if len(point) >= 3 else 0.0
                    compact.append([x, y, z])
        return compact

    @staticmethod
    def _command_vector(world_state: WorldStateSnapshot) -> list[float]:
        raw_vector = world_state.execution.locomotion_proposal_summary.get("command_vector", [])
        if isinstance(raw_vector, list) and len(raw_vector) >= 3:
            try:
                return [float(raw_vector[0]), float(raw_vector[1]), float(raw_vector[2])]
            except (TypeError, ValueError):
                return []
        return []

    @staticmethod
    def _compact_overlay_detections(frame) -> list[dict[str, object]]:
        if frame is None:
            return []
        compact_detections: list[dict[str, object]] = []
        detections = frame.viewer_overlay.get("detections", [])
        if not isinstance(detections, list):
            return []
        for item in detections:
            if not isinstance(item, dict):
                continue
            compact: dict[str, object] = {}
            for key in ("class_name", "track_id"):
                value = item.get(key)
                if isinstance(value, str) and value != "":
                    compact[key] = value
            bbox = SnapshotAdapter._bbox_quad(item.get("bbox_xyxy"))
            if bbox is not None:
                compact["bbox_xyxy"] = bbox
            for key in ("confidence", "depth_m", "approach_yaw_rad"):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    compact[key] = float(value)
            world_pose = SnapshotAdapter._float_triplet(item.get("world_pose_xyz"))
            if world_pose is not None:
                compact["world_pose_xyz"] = world_pose
            compact_detections.append(compact)
        return compact_detections

    @staticmethod
    def selected_target_summary(snapshot: WorldStateSnapshot | None, *, frame=None) -> dict[str, object] | None:
        world_state = WorldStateSnapshot() if snapshot is None else snapshot
        active_target = dict(world_state.execution.active_target)
        if active_target == {}:
            return None

        detections = SnapshotAdapter._compact_overlay_detections(frame)
        track_id = str(active_target.get("target_track_id", active_target.get("track_id", ""))).strip()
        matched_detection = next(
            (
                item
                for item in detections
                if track_id != "" and str(item.get("track_id", "")).strip() == track_id
            ),
            None,
        )
        class_name = str(
            active_target.get(
                "class_name",
                active_target.get("target_class", matched_detection.get("class_name", "") if matched_detection is not None else ""),
            )
        ).strip()
        source = str(active_target.get("source", "")).strip()
        if source == "":
            source = "perception" if matched_detection is not None else "manual"

        payload: dict[str, object] = {
            "className": class_name,
            "trackId": track_id,
            "source": source,
        }

        bbox = SnapshotAdapter._bbox_quad(active_target.get("bbox_xyxy"))
        if bbox is None and matched_detection is not None:
            bbox = SnapshotAdapter._bbox_quad(matched_detection.get("bbox_xyxy"))
        if bbox is not None:
            payload["bbox"] = bbox

        confidence = SnapshotAdapter._optional_float(active_target.get("confidence"))
        if confidence is None and matched_detection is not None:
            confidence = SnapshotAdapter._optional_float(matched_detection.get("confidence"))
        if confidence is not None:
            payload["confidence"] = confidence

        depth_m = SnapshotAdapter._optional_float(active_target.get("depth_m"))
        if depth_m is None and matched_detection is not None:
            depth_m = SnapshotAdapter._optional_float(matched_detection.get("depth_m"))
        if depth_m is not None:
            payload["depthM"] = depth_m

        nav_goal_pixel = SnapshotAdapter._int_pair(active_target.get("nav_goal_pixel"))
        if nav_goal_pixel is not None:
            payload["navGoalPixel"] = nav_goal_pixel

        world_pose = SnapshotAdapter._float_triplet(active_target.get("world_pose_xyz"))
        if world_pose is None:
            world_pose = SnapshotAdapter._float_triplet(active_target.get("world_pose"))
        if world_pose is None and matched_detection is not None:
            world_pose = SnapshotAdapter._float_triplet(matched_detection.get("world_pose_xyz"))
        if world_pose is not None:
            payload["worldPose"] = world_pose

        return payload

    @staticmethod
    def latency_breakdown(
        snapshot: WorldStateSnapshot | None,
        *,
        services: dict[str, object] | None = None,
        transport_state: dict[str, object] | None = None,
    ) -> dict[str, object]:
        world_state = WorldStateSnapshot() if snapshot is None else snapshot
        services = {} if services is None else services
        transport_state = {} if transport_state is None else transport_state
        nav_service = dict(services.get("navdp", {})) if isinstance(services.get("navdp"), dict) else {}
        dual_service = dict(services.get("dual", {})) if isinstance(services.get("dual"), dict) else {}
        detector_report = dict(world_state.perception.detector_runtime_report)
        memory_summary = dict(world_state.memory.summary)
        locomotion_summary = dict(world_state.execution.locomotion_proposal_summary)
        return {
            "frameAgeMs": SnapshotAdapter._optional_float(transport_state.get("frameAgeMs")),
            "perceptionLatencyMs": SnapshotAdapter._optional_float(
                detector_report.get("latency_ms", detector_report.get("latencyMs"))
            ),
            "memoryLatencyMs": SnapshotAdapter._optional_float(
                memory_summary.get("latency_ms", memory_summary.get("latencyMs"))
            ),
            "s2LatencyMs": SnapshotAdapter._optional_float(dual_service.get("latencyMs")),
            "navLatencyMs": SnapshotAdapter._optional_float(nav_service.get("latencyMs")),
            "locomotionLatencyMs": SnapshotAdapter._optional_float(
                locomotion_summary.get("latency_ms", locomotion_summary.get("latencyMs"))
            ),
        }

    @staticmethod
    def cognition_trace_entry(
        snapshot: WorldStateSnapshot | None,
        *,
        system2_output: dict[str, object] | None = None,
        timestamp: float | None = None,
    ) -> dict[str, object]:
        world_state = WorldStateSnapshot() if snapshot is None else snapshot
        selected_target = SnapshotAdapter.selected_target_summary(world_state)
        selected_target_label = ""
        if selected_target is not None:
            track_id = str(selected_target.get("trackId", "")).strip()
            class_name = str(selected_target.get("className", "")).strip()
            selected_target_label = track_id or class_name

        system2_pixel_goal = None
        if world_state.planning.system2_pixel_goal is not None:
            system2_pixel_goal = list(world_state.planning.system2_pixel_goal)

        action_status = dict(world_state.execution.last_action_status)
        return {
            "timestamp": timestamp,
            "frameId": int(world_state.robot.frame_id),
            "taskId": str(world_state.task.task_id),
            "mode": str(world_state.mode),
            "detectionCount": int(world_state.perception.detection_count),
            "trackedDetectionCount": int(world_state.perception.tracked_detection_count),
            "selectedTarget": selected_target_label,
            "memoryObjectCount": int(world_state.memory.object_count),
            "memoryPlaceCount": int(world_state.memory.place_count),
            "s2RawText": "" if system2_output is None else str(system2_output.get("rawText", "")),
            "s2DecisionMode": "" if system2_output is None else str(system2_output.get("decisionMode", "")),
            "s2NeedsRequery": False if system2_output is None else bool(system2_output.get("needsRequery", False)),
            "system2PixelGoal": system2_pixel_goal,
            "planVersion": int(world_state.planning.plan_version),
            "goalVersion": int(world_state.planning.goal_version),
            "trajVersion": int(world_state.planning.traj_version),
            "activeCommandType": str(world_state.execution.active_command_type),
            "actionStatus": str(action_status.get("state", "")),
            "actionReason": str(action_status.get("reason", "")),
            "recoveryState": str(world_state.safety.recovery_state.current_state),
            "recoveryReason": str(world_state.safety.recovery_state.last_trigger_reason),
        }

    @staticmethod
    def to_legacy_runtime_payload(snapshot: WorldStateSnapshot | None) -> dict[str, object]:
        world_state = WorldStateSnapshot() if snapshot is None else snapshot
        nav_trajectory_world = SnapshotAdapter._nav_trajectory_world(world_state)
        command_vector = SnapshotAdapter._command_vector(world_state)
        command_speed_mps = None
        if len(command_vector) >= 2:
            command_speed_mps = math.hypot(command_vector[0], command_vector[1])
        planner = {
            "planVersion": int(world_state.planning.plan_version),
            "goalVersion": int(world_state.planning.goal_version),
            "trajVersion": int(world_state.planning.traj_version),
            "staleSec": float(world_state.planning.stale_info.get("planner_stale_sec", 0.0) or 0.0),
            "executionMode": str(world_state.mode),
            "plannerControlMode": str(world_state.planning.planner_control_mode),
            "plannerControlReason": str(world_state.planning.planner_control_reason),
            "plannerYawDeltaRad": world_state.planning.planner_yaw_delta_rad,
            "activeInstruction": str(world_state.planning.active_instruction),
            "routeState": dict(world_state.planning.route_state),
            "goalDistanceM": world_state.execution.locomotion_proposal_summary.get("goal_distance_m"),
            "yawErrorRad": world_state.execution.locomotion_proposal_summary.get("yaw_error_rad"),
            "navTrajectoryWorld": nav_trajectory_world,
            "navTrajectoryPointCount": int(world_state.planning.active_nav_plan.get("trajectory_point_count", len(nav_trajectory_world)) or 0),
            "commandVector": command_vector,
            "commandSpeedMps": command_speed_mps,
            "actionStatus": None if not world_state.execution.last_action_status else dict(world_state.execution.last_action_status),
            "activeCommandType": str(world_state.execution.active_command_type),
            "globalRouteWaypointIndex": int(world_state.planning.global_route.get("waypoint_index", 0) or 0),
            "globalRouteWaypointCount": int(world_state.planning.global_route.get("waypoint_count", 0) or 0),
            "globalRouteEnabled": bool(world_state.planning.global_route.get("enabled", False)),
            "globalRouteActive": bool(world_state.planning.global_route.get("active", False)),
            "recoveryState": str(world_state.safety.recovery_state.current_state),
            "recoveryEnteredAtNs": int(world_state.safety.recovery_state.entered_at_ns),
            "recoveryRetryCount": int(world_state.safety.recovery_state.retry_count),
            "recoveryBackoffUntilNs": int(world_state.safety.recovery_state.backoff_until_ns),
            "recoveryReason": str(world_state.safety.recovery_state.last_trigger_reason),
        }
        if world_state.planning.system2_pixel_goal is not None:
            planner["system2PixelGoal"] = list(world_state.planning.system2_pixel_goal)
        return {
            "modes": {
                "executionMode": str(world_state.mode),
                "launchMode": str(world_state.runtime.launch_mode),
                "viewerPublish": bool(world_state.runtime.viewer_publish),
                "nativeViewer": str(world_state.runtime.native_viewer),
                "scenePreset": str(world_state.runtime.scene_preset),
                "showDepth": bool(world_state.runtime.show_depth),
                "memoryStore": bool(world_state.runtime.memory_store),
                "detectionEnabled": bool(world_state.runtime.detection_enabled),
            },
            "planner": planner,
            "sensor": {
                "rgbAvailable": bool(world_state.robot.sensor_health.get("observation_available", False)),
                "depthAvailable": bool(world_state.robot.sensor_health.get("batch_available", False)),
                "poseAvailable": world_state.robot.frame_id >= 0,
                "frameId": None if world_state.robot.frame_id < 0 else int(world_state.robot.frame_id),
                "source": "" if world_state.robot.frame_id < 0 else str(world_state.robot.source),
                "cameraPoseXyz": list(world_state.robot.sensor_meta.get("camera_pose_xyz", []))
                if isinstance(world_state.robot.sensor_meta.get("camera_pose_xyz"), list)
                else [],
                "robotPoseXyz": [] if world_state.robot.frame_id < 0 else [float(v) for v in world_state.robot.pose_xyz[:3]],
                "robotYawRad": None if world_state.robot.frame_id < 0 else float(world_state.robot.yaw_rad),
                "sensorMeta": dict(world_state.robot.sensor_meta),
                "captureReport": dict(world_state.robot.capture_report),
                "safeStop": bool(world_state.safety.safe_stop),
                "stale": bool(world_state.safety.stale),
                "timeout": bool(world_state.safety.timeout),
                "sensorUnavailable": bool(world_state.safety.sensor_unavailable),
            },
            "perception": {
                "detectorBackend": str(world_state.perception.detector_backend),
                "detectorSelectedReason": str(world_state.perception.detector_selected_reason),
                "detectorReady": bool(world_state.perception.detector_ready),
                "detectorRuntimeReport": dict(world_state.perception.detector_runtime_report),
                "detectionCount": int(world_state.perception.detection_count),
                "trackedDetectionCount": int(world_state.perception.tracked_detection_count),
                "trajectoryPointCount": int(world_state.perception.trajectory_point_count),
            },
            "memory": {
                "objectCount": int(world_state.memory.object_count),
                "placeCount": int(world_state.memory.place_count),
                "semanticRuleCount": int(world_state.memory.semantic_rule_count),
                "keyframeCount": int(world_state.memory.keyframe_count),
                "scratchpad": dict(world_state.memory.scratchpad),
                "memoryAwareTaskActive": bool(world_state.memory.memory_aware_task_active),
            },
            "transport": {
                "viewerPublish": bool(world_state.runtime.viewer_publish),
                "nativeViewer": str(world_state.runtime.native_viewer),
                "controlEndpoint": str(world_state.runtime.control_endpoint),
                "telemetryEndpoint": str(world_state.runtime.telemetry_endpoint),
                "shmName": str(world_state.runtime.shm_name),
                "frameAvailable": bool(world_state.runtime.frame_available),
                "safeStop": bool(world_state.safety.safe_stop),
                "stale": bool(world_state.safety.stale),
                "timeout": bool(world_state.safety.timeout),
                "sensorUnavailable": bool(world_state.safety.sensor_unavailable),
            },
        }

    @staticmethod
    def to_dashboard_state(
        snapshot: WorldStateSnapshot | None,
        *,
        processes: list[dict[str, object]],
        services: dict[str, object],
        session_state: dict[str, object],
        transport_state: dict[str, object],
        architecture: dict[str, object],
        recent_logs: list[dict[str, object]],
        last_status: dict[str, object] | None = None,
        detector_capability: dict[str, object] | None = None,
        selected_target_summary: dict[str, object] | None = None,
        latency_breakdown: dict[str, object] | None = None,
        cognition_trace: list[dict[str, object]] | None = None,
        recovery_transitions: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        legacy = SnapshotAdapter.to_legacy_runtime_payload(snapshot)
        runtime = dict(legacy["planner"])
        runtime["modes"] = dict(legacy["modes"])
        if isinstance(last_status, dict) and last_status:
            runtime["lastStatusEvent"] = dict(last_status)
        perception = dict(legacy["perception"])
        if isinstance(detector_capability, dict) and detector_capability:
            perception["detectorCapability"] = dict(detector_capability)
        transport = dict(legacy["transport"])
        transport.update(dict(transport_state))
        normalized_selected_target = (
            SnapshotAdapter.selected_target_summary(snapshot)
            if selected_target_summary is None
            else dict(selected_target_summary)
        )
        normalized_latency_breakdown = (
            SnapshotAdapter.latency_breakdown(snapshot, services=services, transport_state=transport_state)
            if latency_breakdown is None
            else dict(latency_breakdown)
        )
        return {
            "timestamp": session_state.get("timestamp"),
            "session": {
                "active": bool(session_state.get("active", False)),
                "startedAt": session_state.get("startedAt"),
                "config": session_state.get("config"),
                "lastEvent": session_state.get("lastEvent"),
            },
            "processes": list(processes),
            "runtime": runtime,
            "sensors": dict(legacy["sensor"]),
            "perception": perception,
            "memory": dict(legacy["memory"]),
            "services": dict(services),
            "architecture": dict(architecture),
            "transport": transport,
            "logs": list(recent_logs),
            "selectedTargetSummary": normalized_selected_target,
            "latencyBreakdown": normalized_latency_breakdown,
            "cognitionTrace": [] if cognition_trace is None else list(cognition_trace),
            "recoveryTransitions": [] if recovery_transitions is None else list(recovery_transitions),
        }

    @staticmethod
    def to_webrtc_state_payload(
        snapshot: WorldStateSnapshot | None,
        *,
        frame,
        has_seen_frame: bool,
        age_ms: float | None,
    ) -> dict[str, object]:
        if frame is None:
            return {
                "type": "waiting_for_frame",
                "age_ms": None if age_ms is None else round(float(age_ms), 3),
                "has_seen_frame": bool(has_seen_frame),
            }
        world_state = WorldStateSnapshot() if snapshot is None else snapshot
        payload = {
            "type": "snapshot",
            "seq": int(frame.seq),
            "frame_id": int(frame.frame_header.frame_id),
            "source": str(frame.frame_header.source),
            "image": {
                "width": int(frame.frame_header.width),
                "height": int(frame.frame_header.height),
                "rgbEncoding": str(frame.frame_header.rgb_encoding),
            },
            "robot_pose_xyz": [float(value) for value in world_state.robot.pose_xyz[:3]],
            "robot_yaw_rad": float(world_state.robot.yaw_rad),
            "sim_time_s": float(frame.frame_header.sim_time_s),
            "detector_backend": str(world_state.perception.detector_backend),
            "detection_count": int(world_state.perception.detection_count),
            "active_command_type": str(world_state.execution.active_command_type),
            "has_depth": frame.depth_image_m is not None,
            "executionMode": str(world_state.mode),
            "planVersion": int(world_state.planning.plan_version),
            "goalVersion": int(world_state.planning.goal_version),
            "trajVersion": int(world_state.planning.traj_version),
            "staleSec": float(world_state.planning.stale_info.get("planner_stale_sec", 0.0) or 0.0),
            "plannerControlMode": str(world_state.planning.planner_control_mode),
            "plannerControlReason": str(world_state.planning.planner_control_reason),
            "plannerYawDeltaRad": world_state.planning.planner_yaw_delta_rad,
            "activeInstruction": str(world_state.planning.active_instruction),
            "routeState": dict(world_state.planning.route_state),
            "recoveryState": str(world_state.safety.recovery_state.current_state),
            "recoveryEnteredAtNs": int(world_state.safety.recovery_state.entered_at_ns),
            "recoveryRetryCount": int(world_state.safety.recovery_state.retry_count),
            "recoveryBackoffUntilNs": int(world_state.safety.recovery_state.backoff_until_ns),
            "recoveryReason": str(world_state.safety.recovery_state.last_trigger_reason),
            "safeStop": bool(world_state.safety.safe_stop),
            "stale": bool(world_state.safety.stale),
            "timeout": bool(world_state.safety.timeout),
            "sensorUnavailable": bool(world_state.safety.sensor_unavailable),
        }
        active_target = dict(world_state.execution.active_target)
        if active_target:
            payload["active_target"] = active_target
            payload["activeTarget"] = dict(active_target)
        if world_state.planning.system2_pixel_goal is not None:
            compact_goal = [int(world_state.planning.system2_pixel_goal[0]), int(world_state.planning.system2_pixel_goal[1])]
            payload["system2_pixel_goal"] = compact_goal
            payload["system2PixelGoal"] = compact_goal
        return payload

    @staticmethod
    def to_webrtc_frame_meta(snapshot: WorldStateSnapshot | None, *, frame) -> dict[str, object] | None:
        if frame is None:
            return None
        world_state = WorldStateSnapshot() if snapshot is None else snapshot
        compact_detections = SnapshotAdapter._compact_overlay_detections(frame)
        compact_trajectory = []
        trajectory_pixels = frame.viewer_overlay.get("trajectory_pixels", [])
        if isinstance(trajectory_pixels, list):
            for point in trajectory_pixels:
                if isinstance(point, list) and len(point) == 2:
                    compact_trajectory.append([int(point[0]), int(point[1])])

        payload: dict[str, object] = {
            "type": "frame_meta",
            "seq": int(frame.seq),
            "frame_id": int(frame.frame_header.frame_id),
            "timestamp_ns": int(frame.frame_header.timestamp_ns),
            "source": str(frame.frame_header.source),
            "robot_pose_xyz": [float(value) for value in world_state.robot.pose_xyz[:3]],
            "robot_yaw_rad": float(world_state.robot.yaw_rad),
            "sim_time_s": float(frame.frame_header.sim_time_s),
            "executionMode": str(world_state.mode),
            "detections": compact_detections,
            "trajectory_pixels": compact_trajectory,
            "trajectoryPixels": compact_trajectory,
            "planVersion": int(world_state.planning.plan_version),
            "goalVersion": int(world_state.planning.goal_version),
            "trajVersion": int(world_state.planning.traj_version),
            "staleSec": float(world_state.planning.stale_info.get("planner_stale_sec", 0.0) or 0.0),
            "plannerControlMode": str(world_state.planning.planner_control_mode),
            "plannerControlReason": str(world_state.planning.planner_control_reason),
            "plannerYawDeltaRad": world_state.planning.planner_yaw_delta_rad,
            "activeInstruction": str(world_state.planning.active_instruction),
            "routeState": dict(world_state.planning.route_state),
            "recoveryState": str(world_state.safety.recovery_state.current_state),
            "recoveryEnteredAtNs": int(world_state.safety.recovery_state.entered_at_ns),
            "recoveryRetryCount": int(world_state.safety.recovery_state.retry_count),
            "recoveryBackoffUntilNs": int(world_state.safety.recovery_state.backoff_until_ns),
            "recoveryReason": str(world_state.safety.recovery_state.last_trigger_reason),
            "safeStop": bool(world_state.safety.safe_stop),
            "stale": bool(world_state.safety.stale),
            "timeout": bool(world_state.safety.timeout),
            "sensorUnavailable": bool(world_state.safety.sensor_unavailable),
        }
        active_target = dict(world_state.execution.active_target)
        if active_target:
            payload["active_target"] = active_target
            payload["activeTarget"] = dict(active_target)
        if world_state.planning.system2_pixel_goal is not None:
            compact_goal = [int(world_state.planning.system2_pixel_goal[0]), int(world_state.planning.system2_pixel_goal[1])]
            payload["system2_pixel_goal"] = compact_goal
            payload["system2PixelGoal"] = compact_goal
        return payload
