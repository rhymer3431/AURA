from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.messages import ActionCommand, ActionStatus
from runtime.planning_session import PlannerStats, TrajectoryUpdate
from runtime.subgoal_executor import CommandEvaluation, SubgoalExecutionResult
from server.command_resolver import CommandResolver


def test_command_resolver_prefers_manual_command_and_preserves_execution_result() -> None:
    resolver = CommandResolver()
    manual = ActionCommand(action_type="LOCAL_SEARCH", task_id="manual", metadata={"planner_managed": True})
    task = ActionCommand(action_type="STOP", task_id="task")
    proposal = resolver.resolve_action_command(manual_command=manual, task_command=task)
    execution = SubgoalExecutionResult(
        command_vector=np.asarray([0.2, 0.0, 0.1], dtype=np.float32),
        trajectory_update=TrajectoryUpdate(
            trajectory_world=np.asarray([[0.1, 0.0, 0.0]], dtype=np.float32),
            plan_version=1,
            stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
            source_frame_id=1,
        ),
        evaluation=CommandEvaluation(force_stop=False, goal_distance_m=0.7, yaw_error_rad=0.0, reached_goal=False),
        status=ActionStatus(command_id=manual.command_id, state="running"),
    )

    resolved = resolver.resolve_execution(proposal=proposal, execution=execution)

    assert proposal.source == "manual"
    assert proposal.action_command is manual
    assert resolved.action_command is manual
    assert tuple(round(float(v), 4) for v in resolved.command_vector.tolist()) == (0.2, 0.0, 0.1)
    assert resolved.status is not None and resolved.status.state == "running"
