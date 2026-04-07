from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ProposalPreparationResult:
    task_command: object | None
    proposal: object


@dataclass(frozen=True)
class SafetyResolutionResult:
    resolved: object
    recovery_state: object


class SafetyResolutionService:
    def __init__(
        self,
        *,
        task_command_client,
        command_resolver,
        safety_supervisor,
        task_manager,
        world_state_port,
        memory_client,
        planner_coordinator,
    ) -> None:  # noqa: ANN001
        self._task_command_client = task_command_client
        self._command_resolver = command_resolver
        self._safety_supervisor = safety_supervisor
        self._task_manager = task_manager
        self._world_state = world_state_port
        self._memory_client = memory_client
        self._planner_coordinator = planner_coordinator

    @property
    def memory_client(self):  # noqa: ANN201
        return self._memory_client

    def prepare_proposal(
        self,
        *,
        frame_event,
        directive,
        runtime_status,
        manual_command,
    ) -> ProposalPreparationResult:  # noqa: ANN001
        task_command = None
        if directive.route_task_command:
            task_command = self._task_command_client.step(
                now=time.time(),
                robot_pose=tuple(float(v) for v in frame_event.robot_pose_xyz[:3]),
                robot_yaw_rad=float(frame_event.robot_yaw_rad),
                action_status=runtime_status,
                publish=False,
            )
        proposal = self._command_resolver.resolve_action_command(
            manual_command=manual_command,
            task_command=task_command,
        )
        return ProposalPreparationResult(task_command=task_command, proposal=proposal)

    def resolve(
        self,
        *,
        frame_event,
        execution,
        proposal,
        planning_evaluation,
        now_ns: int,
    ) -> SafetyResolutionResult:  # noqa: ANN001
        safety_decision = self._safety_supervisor.evaluate(
            frame_event=frame_event,
            trajectory_update=execution.trajectory_update,
            recovery_state=planning_evaluation.recovery_state,
            now_ns=now_ns,
        )
        final_recovery = safety_decision.recovery_state
        self._world_state.set_recovery_state(final_recovery)
        resolved = self._command_resolver.resolve_execution(
            proposal=proposal,
            execution=execution,
            recovery_state=final_recovery,
            safety_decision=safety_decision,
        )
        task_reset = self._task_manager.sync_after_update(
            resolved.trajectory_update,
            resolved.status,
            memory_client=self._memory_client,
        )
        self._world_state.update_task(self._task_manager.snapshot())
        if task_reset:
            self._world_state.reset_recovery_state(entered_at_ns=now_ns, reason="task_reset")
            final_recovery = self._world_state.recovery_state()
            resolved = self._command_resolver.resolve_execution(
                proposal=proposal,
                execution=execution,
                recovery_state=final_recovery,
                safety_decision=safety_decision,
            )
        self._world_state.record_planning_result(
            resolved.trajectory_update,
            planner_state=self._planner_coordinator.runtime_state,
            recovery_state=final_recovery,
        )
        self._world_state.record_command_decision(resolved, recovery_state=final_recovery)
        return SafetyResolutionResult(resolved=resolved, recovery_state=final_recovery)
