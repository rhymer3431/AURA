from __future__ import annotations

from dataclasses import dataclass

from schemas.events import FrameEvent
from schemas.execution_mode import normalize_execution_mode
from schemas.recovery import RecoveryState, RecoveryStateSnapshot
from schemas.world_state import TaskSnapshot, WorldStateSnapshot

from .recovery_state import RecoveryEvent, RecoveryPolicy, apply_recovery_event, is_terminal_recovery_state


@dataclass(frozen=True)
class DecisionDirective:
    process_perception: bool
    retrieve_memory: bool
    use_manual_command: bool
    route_task_command: bool
    allow_planning: bool
    backoff_active: bool
    reason: str = ""


@dataclass(frozen=True)
class DualDecisionDirective:
    force_s2: bool
    launch_s2: bool
    launch_s1: bool
    goal_missing: bool
    goal_stale: bool
    traj_missing: bool
    traj_stale: bool
    discard_stale_traj: bool
    backoff_active: bool


@dataclass(frozen=True)
class RecoveryEvaluation:
    recovery_state: RecoveryStateSnapshot
    event: RecoveryEvent | None = None
    reason: str = ""


class DecisionEngine:
    def __init__(self, policy: RecoveryPolicy | None = None) -> None:
        self._policy = policy

    def evaluate(
        self,
        *,
        world_state: WorldStateSnapshot,
        task: TaskSnapshot,
        frame_event: FrameEvent,
        manual_command_present: bool,
        active_memory_instruction: str,
        recovery_state: RecoveryStateSnapshot | None = None,
        now_ns: int | None = None,
    ) -> DecisionDirective:
        current = world_state.recovery_state if recovery_state is None else recovery_state
        execution_mode = normalize_execution_mode(task.mode or world_state.mode)
        current_state = current.state
        current_time_ns = int(frame_event.timestamp_ns if now_ns is None else now_ns)
        backoff_active = bool(current.backoff_until_ns > 0 and current_time_ns < int(current.backoff_until_ns))
        allow_planning = execution_mode in {"NAV", "MEM_NAV", "EXPLORE"} and current_state not in {
            RecoveryState.WAIT_SENSOR,
            RecoveryState.SAFE_STOP,
            RecoveryState.FAILED,
        }
        if backoff_active:
            allow_planning = False
        process_perception = bool(frame_event.batch is not None and frame_event.observation is not None and execution_mode in {"NAV", "MEM_NAV", "EXPLORE"})
        retrieve_memory = False
        route_task_command = False
        reason = f"mode:{execution_mode}"
        if not process_perception:
            reason = "planning_suppressed"
        if is_terminal_recovery_state(current_state):
            reason = f"recovery:{current_state.value}"
        elif backoff_active:
            reason = "recovery_backoff"
        elif current_state == RecoveryState.WAIT_SENSOR:
            reason = "waiting_for_sensor"
        return DecisionDirective(
            process_perception=process_perception,
            retrieve_memory=retrieve_memory,
            use_manual_command=bool(manual_command_present),
            route_task_command=route_task_command,
            allow_planning=allow_planning,
            backoff_active=backoff_active,
            reason=reason,
        )

    def evaluate_planning_outcome(
        self,
        *,
        world_state: WorldStateSnapshot,
        task: TaskSnapshot,
        recovery_state: RecoveryStateSnapshot,
        trajectory_update,
        action_command,
        now_ns: int,
    ) -> RecoveryEvaluation:  # noqa: ANN001
        if self._policy is None:
            return RecoveryEvaluation(recovery_state=recovery_state, reason="recovery_policy_unconfigured")
        if task.state in {"idle", "cancelled", "completed"} or action_command is None:
            return RecoveryEvaluation(
                recovery_state=apply_recovery_event(
                    recovery_state,
                    RecoveryEvent(kind="task_reset", reason="task_reset"),
                    now_ns=now_ns,
                    policy=self._policy,
                ),
                event=RecoveryEvent(kind="task_reset", reason="task_reset"),
                reason="task_reset",
            )

        if self._is_fresh_planning_success(world_state=world_state, trajectory_update=trajectory_update):
            event = RecoveryEvent(kind="planning_success", reason="planning_success")
            return RecoveryEvaluation(
                recovery_state=apply_recovery_event(recovery_state, event, now_ns=now_ns, policy=self._policy),
                event=event,
                reason="planning_success",
            )

        last_error = str(getattr(getattr(trajectory_update, "stats", None), "last_error", "") or "").strip()
        planner_mode = str(getattr(trajectory_update, "planner_control_mode", "") or "")
        if last_error != "" and planner_mode not in {"stop", "wait", "yaw_delta"}:
            event = RecoveryEvent(kind="planning_failure", reason=last_error)
            return RecoveryEvaluation(
                recovery_state=apply_recovery_event(recovery_state, event, now_ns=now_ns, policy=self._policy),
                event=event,
                reason="planning_failure",
            )
        return RecoveryEvaluation(recovery_state=recovery_state, reason="planning_unchanged")

    @staticmethod
    def _is_fresh_planning_success(*, world_state: WorldStateSnapshot, trajectory_update) -> bool:  # noqa: ANN001
        last_error = str(getattr(getattr(trajectory_update, "stats", None), "last_error", "") or "").strip()
        if last_error != "":
            return False
        planner_mode = str(getattr(trajectory_update, "planner_control_mode", "") or "")
        if planner_mode in {"stop", "wait", "yaw_delta"}:
            return True
        plan_version = int(getattr(trajectory_update, "plan_version", -1))
        goal_version = int(getattr(trajectory_update, "goal_version", -1))
        traj_version = int(getattr(trajectory_update, "traj_version", -1))
        return bool(
            plan_version > int(world_state.planning.plan_version)
            or goal_version > int(world_state.planning.goal_version)
            or traj_version > int(world_state.planning.traj_version)
        )

    def evaluate_dual(
        self,
        *,
        now: float,
        goal_cache,
        traj_cache,
        last_s1_ts: float,
        last_s2_ts: float,
        s1_period_sec: float,
        s2_period_sec: float,
        goal_ttl_sec: float,
        traj_ttl_sec: float,
        traj_max_stale_sec: float,
        s2_retry_after_ts: float,
        force_s2_pending: bool,
        events: dict[str, object] | None = None,
    ) -> DualDecisionDirective:
        goal = goal_cache
        traj = traj_cache
        event_map = dict(events or {})
        external_force = bool(event_map.get("force_s2", False)) or bool(event_map.get("stuck", False)) or bool(
            event_map.get("collision_risk", False)
        )
        force_s2 = bool(external_force or force_s2_pending)
        goal_missing = goal is None
        goal_stale = bool(goal is not None and (now - float(goal.updated_at)) > float(goal_ttl_sec))
        due_s2 = (now - float(last_s2_ts)) >= float(s2_period_sec)
        awaiting_first_traj = bool(goal is not None and traj is None and str(goal.mode) == "pixel_goal" and not bool(goal.stop))
        backoff_active = now < float(s2_retry_after_ts)
        should_s2 = bool(force_s2 or goal_missing or goal_stale or due_s2)
        if awaiting_first_traj and not force_s2:
            should_s2 = bool(goal_missing)
        if backoff_active and not force_s2:
            should_s2 = False

        goal_is_pixel = bool(
            goal is not None
            and str(goal.mode) == "pixel_goal"
            and getattr(goal, "pixel_x", None) is not None
            and getattr(goal, "pixel_y", None) is not None
            and not bool(goal.stop)
        )
        goal_changed = bool(goal_is_pixel and (traj is None or int(traj.goal_version) != int(goal.version)))
        traj_missing = traj is None
        traj_stale = bool(traj is not None and (now - float(traj.updated_at)) > float(traj_ttl_sec))
        discard_stale_traj = bool(traj is not None and (now - float(traj.updated_at)) > float(traj_max_stale_sec))
        due_s1 = (now - float(last_s1_ts)) >= float(s1_period_sec)
        should_s1 = bool(goal_is_pixel and (goal_changed or traj_missing or traj_stale or due_s1))
        return DualDecisionDirective(
            force_s2=force_s2,
            launch_s2=should_s2,
            launch_s1=should_s1,
            goal_missing=goal_missing,
            goal_stale=goal_stale,
            traj_missing=traj_missing,
            traj_stale=traj_stale,
            discard_stale_traj=discard_stale_traj,
            backoff_active=backoff_active,
        )
