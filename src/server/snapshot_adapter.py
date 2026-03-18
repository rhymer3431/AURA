from __future__ import annotations

from typing import Any

from schemas.world_state import WorldStateSnapshot


class SnapshotAdapter:
    @staticmethod
    def to_legacy_runtime_payload(snapshot: WorldStateSnapshot | None) -> dict[str, object]:
        world_state = WorldStateSnapshot() if snapshot is None else snapshot
        planner = {
            "planVersion": int(world_state.planning.plan_version),
            "goalVersion": int(world_state.planning.goal_version),
            "trajVersion": int(world_state.planning.traj_version),
            "staleSec": float(world_state.planning.stale_info.get("planner_stale_sec", 0.0) or 0.0),
            "plannerControlMode": str(world_state.planning.planner_control_mode),
            "plannerControlReason": str(world_state.planning.planner_control_reason),
            "plannerYawDeltaRad": world_state.planning.planner_yaw_delta_rad,
            "goalDistanceM": world_state.execution.locomotion_proposal_summary.get("goal_distance_m"),
            "yawErrorRad": world_state.execution.locomotion_proposal_summary.get("yaw_error_rad"),
            "interactivePhase": str(world_state.planning.interactive_phase),
            "interactiveCommandId": int(world_state.planning.interactive_command_id),
            "interactiveInstruction": str(world_state.planning.interactive_instruction),
            "actionStatus": None if not world_state.execution.last_action_status else dict(world_state.execution.last_action_status),
            "activeCommandType": str(world_state.execution.active_command_type),
            "globalRouteWaypointIndex": int(world_state.planning.global_route.get("waypoint_index", 0) or 0),
            "globalRouteWaypointCount": int(world_state.planning.global_route.get("waypoint_count", 0) or 0),
            "globalRouteEnabled": bool(world_state.planning.global_route.get("enabled", False)),
            "globalRouteActive": bool(world_state.planning.global_route.get("active", False)),
        }
        if world_state.planning.system2_pixel_goal is not None:
            planner["system2PixelGoal"] = list(world_state.planning.system2_pixel_goal)
        return {
            "modes": {
                "plannerMode": str(world_state.mode),
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
        recent_logs: list[dict[str, object]],
        last_status: dict[str, object] | None = None,
        detector_capability: dict[str, object] | None = None,
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
            "transport": transport,
            "logs": list(recent_logs),
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
            "planVersion": int(world_state.planning.plan_version),
            "goalVersion": int(world_state.planning.goal_version),
            "trajVersion": int(world_state.planning.traj_version),
            "staleSec": float(world_state.planning.stale_info.get("planner_stale_sec", 0.0) or 0.0),
            "plannerControlMode": str(world_state.planning.planner_control_mode),
            "plannerYawDeltaRad": world_state.planning.planner_yaw_delta_rad,
            "interactivePhase": str(world_state.planning.interactive_phase),
            "interactiveCommandId": int(world_state.planning.interactive_command_id),
            "interactiveInstruction": str(world_state.planning.interactive_instruction),
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
        compact_detections: list[dict[str, object]] = []
        detections = frame.viewer_overlay.get("detections", [])
        if isinstance(detections, list):
            for item in detections:
                if not isinstance(item, dict):
                    continue
                compact: dict[str, object] = {}
                for key in ("class_name", "track_id"):
                    value = item.get(key)
                    if isinstance(value, str) and value != "":
                        compact[key] = value
                bbox = item.get("bbox_xyxy")
                if isinstance(bbox, list) and len(bbox) == 4:
                    compact["bbox_xyxy"] = [int(value) for value in bbox]
                for key in ("confidence", "depth_m", "approach_yaw_rad"):
                    value = item.get(key)
                    if isinstance(value, (int, float)):
                        compact[key] = float(value)
                world_pose = item.get("world_pose_xyz")
                if isinstance(world_pose, list) and len(world_pose) >= 3:
                    compact["world_pose_xyz"] = [float(world_pose[0]), float(world_pose[1]), float(world_pose[2])]
                compact_detections.append(compact)
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
            "detections": compact_detections,
            "trajectory_pixels": compact_trajectory,
            "trajectoryPixels": compact_trajectory,
            "planVersion": int(world_state.planning.plan_version),
            "goalVersion": int(world_state.planning.goal_version),
            "trajVersion": int(world_state.planning.traj_version),
            "staleSec": float(world_state.planning.stale_info.get("planner_stale_sec", 0.0) or 0.0),
            "plannerControlMode": str(world_state.planning.planner_control_mode),
            "plannerYawDeltaRad": world_state.planning.planner_yaw_delta_rad,
            "interactivePhase": str(world_state.planning.interactive_phase),
            "interactiveCommandId": int(world_state.planning.interactive_command_id),
            "interactiveInstruction": str(world_state.planning.interactive_instruction),
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
