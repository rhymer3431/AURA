from __future__ import annotations

import numpy as np

from ipc.messages import ActionCommand
from runtime.subgoal_executor import SubgoalExecutionResult
from schemas.commands import CommandProposal, ResolvedCommand


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
        execution: SubgoalExecutionResult,
    ) -> ResolvedCommand:
        return ResolvedCommand(
            action_command=proposal.action_command,
            command_vector=np.asarray(execution.command_vector, dtype=np.float32).copy(),
            trajectory_update=execution.trajectory_update,
            evaluation=execution.evaluation,
            status=execution.status,
            source=str(proposal.source),
            safety_override=False,
            metadata=dict(proposal.metadata),
        )
