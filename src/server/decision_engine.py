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
