from __future__ import annotations

import numpy as np

from ipc.messages import ActionCommand, ActionStatus
from schemas.recovery import RecoveryState, RecoveryStateSnapshot
from schemas.commands import CommandProposal, LocomotionProposal, ResolvedCommand

from .safety_supervisor import SafetyDecision


class CommandResolver:
    def resolve_action_command(
        self,
        *,
        manual_command: ActionCommand | None,
        task_command: ActionCommand | None,
    ) -> CommandProposal:
        if manual_command is not None:
            return CommandProposal(
                source="manual",
                priority=100,
                action_command=manual_command,
            )
        return CommandProposal(
            source="task_orchestrator",
            priority=10,
            action_command=task_command,
        )

    def resolve_execution(
        self,
        *,
        proposal: CommandProposal,
        execution: LocomotionProposal,
        recovery_state: RecoveryStateSnapshot,
        safety_decision: SafetyDecision | None = None,
    ) -> ResolvedCommand:
        current_state = recovery_state.state
        metadata = {
            **dict(proposal.metadata),
            **dict(execution.metadata),
            "recovery_state": recovery_state.to_dict(),
            "recovery_reason": str(recovery_state.last_trigger_reason),
            "retry_count": int(recovery_state.retry_count),
            "backoff_until_ns": int(recovery_state.backoff_until_ns),
        }
        safety_override = bool(safety_decision.safety_override) if safety_decision is not None else False
        command_vector = np.asarray(execution.command_vector, dtype=np.float32).copy()
        status: ActionStatus | None
        if current_state == RecoveryState.NORMAL:
            status = self._build_status(
                action_command=proposal.action_command,
                execution=execution,
                extra_metadata=metadata,
            )
        elif current_state == RecoveryState.FAILED:
            command_vector = np.zeros(3, dtype=np.float32)
            safety_override = True
            status = self._recovery_status(
                action_command=proposal.action_command,
                state="failed",
                reason=str(recovery_state.last_trigger_reason or "recovery_failed"),
                metadata=metadata,
            )
        elif current_state == RecoveryState.RECOVERY_TURN and execution.trajectory_update.planner_control_mode == "yaw_delta":
            status = self._build_status(
                action_command=proposal.action_command,
                execution=execution,
                extra_metadata=metadata,
            )
        else:
            command_vector = np.zeros(3, dtype=np.float32)
            safety_override = safety_override or current_state in {RecoveryState.WAIT_SENSOR, RecoveryState.SAFE_STOP}
            status = self._recovery_status(
                action_command=proposal.action_command,
                state="stale",
                reason=str(recovery_state.last_trigger_reason or current_state.value.lower()),
                metadata=metadata,
            )
        return ResolvedCommand(
            action_command=proposal.action_command,
            command_vector=command_vector,
            trajectory_update=execution.trajectory_update,
            evaluation=execution.evaluation,
            status=status,
            source=str(proposal.source),
            safety_override=safety_override,
            metadata=metadata,
        )

    def _build_status(
        self,
        *,
        action_command: ActionCommand | None,
        execution: LocomotionProposal,
        extra_metadata: dict[str, object],
    ) -> ActionStatus | None:
        if action_command is None:
            return None
        trajectory_update = execution.trajectory_update
        evaluation = execution.evaluation
        metadata = {**dict(extra_metadata), **dict(execution.metadata)}
        planner_managed = bool(action_command.metadata.get("planner_managed", False))
        if action_command.action_type == "STOP":
            return ActionStatus(
                command_id=action_command.command_id,
                state="succeeded",
                success=True,
                distance_remaining_m=0.0,
                metadata={"action_type": action_command.action_type, **metadata},
            )
        if evaluation.reached_goal:
            return ActionStatus(
                command_id=action_command.command_id,
                state="succeeded",
                success=True,
                distance_remaining_m=max(float(evaluation.goal_distance_m), 0.0),
                metadata={"action_type": action_command.action_type, "planner_managed": planner_managed, **metadata},
            )
        if planner_managed and bool(trajectory_update.stop) and trajectory_update.stats.last_error == "":
            return ActionStatus(
                command_id=action_command.command_id,
                state="succeeded",
                success=True,
                distance_remaining_m=0.0,
                metadata={"action_type": action_command.action_type, "planner_managed": True, **metadata},
            )
        if (
            trajectory_update.stats.last_error != ""
            and trajectory_update.trajectory_world.shape[0] == 0
            and action_command.action_type not in {"STOP", "LOOK_AT"}
        ):
            return ActionStatus(
                command_id=action_command.command_id,
                state="failed",
                success=False,
                reason=trajectory_update.stats.last_error,
                distance_remaining_m=None if evaluation.goal_distance_m < 0.0 else float(evaluation.goal_distance_m),
                metadata={"action_type": action_command.action_type, "planner_managed": planner_managed, **metadata},
            )
        if action_command.action_type == "LOOK_AT" and abs(float(evaluation.yaw_error_rad)) < 0.05:
            return ActionStatus(
                command_id=action_command.command_id,
                state="succeeded",
                success=True,
                distance_remaining_m=0.0,
                metadata={"action_type": action_command.action_type, **metadata},
            )
        return ActionStatus(
            command_id=action_command.command_id,
            state="running",
            success=False,
            distance_remaining_m=None if evaluation.goal_distance_m < 0.0 else float(evaluation.goal_distance_m),
            metadata={"action_type": action_command.action_type, "planner_managed": planner_managed, **metadata},
        )

    @staticmethod
    def _recovery_status(
        *,
        action_command: ActionCommand | None,
        state: str,
        reason: str,
        metadata: dict[str, object],
    ) -> ActionStatus | None:
        if action_command is None:
            return None
        return ActionStatus(
            command_id=action_command.command_id,
            state=state,
            success=bool(state == "succeeded"),
            reason=str(reason),
            metadata={"action_type": action_command.action_type, **metadata},
        )
