from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from schemas.recovery import RecoveryStateSnapshot
from schemas.execution_mode import ExecutionMode, normalize_execution_mode


def _dict(payload: object) -> dict[str, Any]:
    return dict(payload) if isinstance(payload, dict) else {}


def _list(payload: object) -> list[Any]:
    return list(payload) if isinstance(payload, list) else []


def _pose_xyz(payload: object) -> tuple[float, float, float]:
    if isinstance(payload, (list, tuple)) and len(payload) >= 3:
        return (float(payload[0]), float(payload[1]), float(payload[2]))
    return (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str = ""
    instruction: str = ""
    mode: ExecutionMode = "IDLE"
    state: str = "idle"
    command_id: int = -1

    @classmethod
    def from_dict(cls, payload: object) -> TaskSnapshot:
        data = _dict(payload)
        return cls(
            task_id=str(data.get("task_id", "")),
            instruction=str(data.get("instruction", "")),
            mode=normalize_execution_mode(data.get("mode")),
            state=str(data.get("state", "idle")),
            command_id=int(data.get("command_id", -1) or -1),
        )


@dataclass(frozen=True)
class RobotStateSnapshot:
    pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_rad: float = 0.0
    frame_id: int = -1
    timestamp_ns: int = 0
    source: str = ""
    sensor_health: dict[str, Any] = field(default_factory=dict)
    sensor_meta: dict[str, Any] = field(default_factory=dict)
    capture_report: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: object) -> RobotStateSnapshot:
        data = _dict(payload)
        return cls(
            pose_xyz=_pose_xyz(data.get("pose_xyz")),
            yaw_rad=float(data.get("yaw_rad", 0.0) or 0.0),
            frame_id=int(data.get("frame_id", -1) or -1),
            timestamp_ns=int(data.get("timestamp_ns", 0) or 0),
            source=str(data.get("source", "")),
            sensor_health=_dict(data.get("sensor_health")),
            sensor_meta=_dict(data.get("sensor_meta")),
            capture_report=_dict(data.get("capture_report")),
        )


@dataclass(frozen=True)
class PerceptionStateSnapshot:
    summary: dict[str, Any] = field(default_factory=dict)
    detector_backend: str = ""
    detector_selected_reason: str = ""
    detector_ready: bool = False
    detector_runtime_report: dict[str, Any] = field(default_factory=dict)
    detection_count: int = 0
    tracked_detection_count: int = 0
    trajectory_point_count: int = 0

    @classmethod
    def from_dict(cls, payload: object) -> PerceptionStateSnapshot:
        data = _dict(payload)
        return cls(
            summary=_dict(data.get("summary")),
            detector_backend=str(data.get("detector_backend", "")),
            detector_selected_reason=str(data.get("detector_selected_reason", "")),
            detector_ready=bool(data.get("detector_ready", False)),
            detector_runtime_report=_dict(data.get("detector_runtime_report")),
            detection_count=int(data.get("detection_count", 0) or 0),
            tracked_detection_count=int(data.get("tracked_detection_count", 0) or 0),
            trajectory_point_count=int(data.get("trajectory_point_count", 0) or 0),
        )


@dataclass(frozen=True)
class MemoryStateSnapshot:
    summary: dict[str, Any] = field(default_factory=dict)
    object_count: int = 0
    place_count: int = 0
    semantic_rule_count: int = 0
    keyframe_count: int = 0
    scratchpad: dict[str, Any] = field(default_factory=dict)
    memory_aware_task_active: bool = False

    @classmethod
    def from_dict(cls, payload: object) -> MemoryStateSnapshot:
        data = _dict(payload)
        return cls(
            summary=_dict(data.get("summary")),
            object_count=int(data.get("object_count", 0) or 0),
            place_count=int(data.get("place_count", 0) or 0),
            semantic_rule_count=int(data.get("semantic_rule_count", 0) or 0),
            keyframe_count=int(data.get("keyframe_count", 0) or 0),
            scratchpad=_dict(data.get("scratchpad")),
            memory_aware_task_active=bool(data.get("memory_aware_task_active", False)),
        )


@dataclass(frozen=True)
class PlanningStateSnapshot:
    last_s2_result: dict[str, Any] = field(default_factory=dict)
    active_nav_plan: dict[str, Any] = field(default_factory=dict)
    plan_version: int = -1
    goal_version: int = -1
    traj_version: int = -1
    planner_mode: ExecutionMode = "IDLE"
    active_instruction: str = ""
    route_state: dict[str, Any] = field(default_factory=dict)
    planner_control_mode: str = ""
    planner_control_reason: str = ""
    planner_yaw_delta_rad: float | None = None
    system2_pixel_goal: list[int] | None = None
    stale_info: dict[str, Any] = field(default_factory=dict)
    global_route: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: object) -> PlanningStateSnapshot:
        data = _dict(payload)
        raw_goal = data.get("system2_pixel_goal")
        system2_pixel_goal = None
        if isinstance(raw_goal, list) and len(raw_goal) >= 2:
            system2_pixel_goal = [int(raw_goal[0]), int(raw_goal[1])]
        return cls(
            last_s2_result=_dict(data.get("last_s2_result")),
            active_nav_plan=_dict(data.get("active_nav_plan")),
            plan_version=int(data.get("plan_version", -1) or -1),
            goal_version=int(data.get("goal_version", -1) or -1),
            traj_version=int(data.get("traj_version", -1) or -1),
            planner_mode=normalize_execution_mode(data.get("planner_mode")),
            active_instruction=str(data.get("active_instruction", data.get("interactive_instruction", ""))),
            route_state=_dict(data.get("route_state")),
            planner_control_mode=str(data.get("planner_control_mode", "")),
            planner_control_reason=str(data.get("planner_control_reason", "")),
            planner_yaw_delta_rad=None
            if data.get("planner_yaw_delta_rad") is None
            else float(data.get("planner_yaw_delta_rad", 0.0)),
            system2_pixel_goal=system2_pixel_goal,
            stale_info=_dict(data.get("stale_info")),
            global_route=_dict(data.get("global_route")),
        )


@dataclass(frozen=True)
class ExecutionStateSnapshot:
    last_command_decision: dict[str, Any] = field(default_factory=dict)
    last_action_status: dict[str, Any] = field(default_factory=dict)
    active_overrides: dict[str, Any] = field(default_factory=dict)
    locomotion_proposal_summary: dict[str, Any] = field(default_factory=dict)
    active_command_type: str = ""
    active_target: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: object) -> ExecutionStateSnapshot:
        data = _dict(payload)
        return cls(
            last_command_decision=_dict(data.get("last_command_decision")),
            last_action_status=_dict(data.get("last_action_status")),
            active_overrides=_dict(data.get("active_overrides")),
            locomotion_proposal_summary=_dict(data.get("locomotion_proposal_summary")),
            active_command_type=str(data.get("active_command_type", "")),
            active_target=_dict(data.get("active_target")),
        )


@dataclass(frozen=True)
class SafetyStateSnapshot:
    safe_stop: bool = False
    stale: bool = False
    timeout: bool = False
    sensor_unavailable: bool = False
    recovery_state: RecoveryStateSnapshot = field(default_factory=RecoveryStateSnapshot)

    @classmethod
    def from_dict(cls, payload: object) -> SafetyStateSnapshot:
        data = _dict(payload)
        return cls(
            safe_stop=bool(data.get("safe_stop", False)),
            stale=bool(data.get("stale", False)),
            timeout=bool(data.get("timeout", False)),
            sensor_unavailable=bool(data.get("sensor_unavailable", False)),
            recovery_state=RecoveryStateSnapshot.from_dict(data.get("recovery_state")),
        )


@dataclass(frozen=True)
class RuntimeStateSnapshot:
    launch_mode: str = ""
    viewer_publish: bool = False
    native_viewer: str = ""
    scene_preset: str = ""
    show_depth: bool = False
    memory_store: bool = False
    detection_enabled: bool = True
    control_endpoint: str = ""
    telemetry_endpoint: str = ""
    shm_name: str = ""
    frame_available: bool = False

    @classmethod
    def from_dict(cls, payload: object) -> RuntimeStateSnapshot:
        data = _dict(payload)
        return cls(
            launch_mode=str(data.get("launch_mode", "")),
            viewer_publish=bool(data.get("viewer_publish", False)),
            native_viewer=str(data.get("native_viewer", "")),
            scene_preset=str(data.get("scene_preset", "")),
            show_depth=bool(data.get("show_depth", False)),
            memory_store=bool(data.get("memory_store", False)),
            detection_enabled=bool(data.get("detection_enabled", True)),
            control_endpoint=str(data.get("control_endpoint", "")),
            telemetry_endpoint=str(data.get("telemetry_endpoint", "")),
            shm_name=str(data.get("shm_name", "")),
            frame_available=bool(data.get("frame_available", False)),
        )


@dataclass(frozen=True)
class WorldStateSnapshot:
    task: TaskSnapshot = field(default_factory=TaskSnapshot)
    mode: ExecutionMode = "IDLE"
    robot: RobotStateSnapshot = field(default_factory=RobotStateSnapshot)
    perception: PerceptionStateSnapshot = field(default_factory=PerceptionStateSnapshot)
    memory: MemoryStateSnapshot = field(default_factory=MemoryStateSnapshot)
    planning: PlanningStateSnapshot = field(default_factory=PlanningStateSnapshot)
    execution: ExecutionStateSnapshot = field(default_factory=ExecutionStateSnapshot)
    safety: SafetyStateSnapshot = field(default_factory=SafetyStateSnapshot)
    runtime: RuntimeStateSnapshot = field(default_factory=RuntimeStateSnapshot)

    @property
    def current_task(self) -> TaskSnapshot:
        return self.task

    @property
    def robot_pose_xyz(self) -> tuple[float, float, float]:
        return self.robot.pose_xyz

    @property
    def robot_yaw_rad(self) -> float:
        return self.robot.yaw_rad

    @property
    def last_perception_summary(self) -> dict[str, Any]:
        return dict(self.perception.summary)

    @property
    def last_memory_context(self) -> dict[str, Any]:
        return dict(self.memory.summary)

    @property
    def last_s2_result(self) -> dict[str, Any]:
        return dict(self.planning.last_s2_result)

    @property
    def active_nav_plan(self) -> dict[str, Any]:
        return dict(self.planning.active_nav_plan)

    @property
    def recovery_state(self) -> RecoveryStateSnapshot:
        return RecoveryStateSnapshot.from_dict(self.safety.recovery_state.to_dict())

    @property
    def stale_timers(self) -> dict[str, Any]:
        return dict(self.planning.stale_info)

    @property
    def last_command_decision(self) -> dict[str, Any]:
        return dict(self.execution.last_command_decision)

    @property
    def sensor_health(self) -> dict[str, Any]:
        return dict(self.robot.sensor_health)

    @property
    def active_overrides(self) -> dict[str, Any]:
        return dict(self.execution.active_overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> WorldStateSnapshot:
        data = _dict(payload)
        return cls(
            task=TaskSnapshot.from_dict(data.get("task")),
            mode=normalize_execution_mode(data.get("mode")),
            robot=RobotStateSnapshot.from_dict(data.get("robot")),
            perception=PerceptionStateSnapshot.from_dict(data.get("perception")),
            memory=MemoryStateSnapshot.from_dict(data.get("memory")),
            planning=PlanningStateSnapshot.from_dict(data.get("planning")),
            execution=ExecutionStateSnapshot.from_dict(data.get("execution")),
            safety=SafetyStateSnapshot.from_dict(data.get("safety")),
            runtime=RuntimeStateSnapshot.from_dict(data.get("runtime")),
        )
