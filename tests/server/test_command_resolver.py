from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.messages import ActionCommand, ActionStatus
from locomotion.types import CommandEvaluation
from runtime.planning_session import PlannerStats, TrajectoryUpdate
from schemas.commands import LocomotionProposal
from schemas.recovery import RecoveryState, RecoveryStateSnapshot
from server.command_resolver import CommandResolver
from server.safety_supervisor import SafetyDecision


def test_command_resolver_prefers_manual_command_and_preserves_execution_result() -> None:
    resolver = CommandResolver()
    manual = ActionCommand(action_type="LOCAL_SEARCH", task_id="manual", metadata={"planner_managed": True})
    task = ActionCommand(action_type="STOP", task_id="task")
    proposal = resolver.resolve_action_command(manual_command=manual, task_command=task)
    execution = LocomotionProposal(
        command_vector=np.asarray([0.2, 0.0, 0.1], dtype=np.float32),
        trajectory_update=TrajectoryUpdate(
            trajectory_world=np.asarray([[0.1, 0.0, 0.0]], dtype=np.float32),
            plan_version=1,
            stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
            source_frame_id=1,
        ),
        evaluation=CommandEvaluation(force_stop=False, goal_distance_m=0.7, yaw_error_rad=0.0, reached_goal=False),
        metadata={"reason": "unit"},
    )

    resolved = resolver.resolve_execution(
        proposal=proposal,
        execution=execution,
        recovery_state=RecoveryStateSnapshot.normal(),
        safety_decision=SafetyDecision(recovery_state=RecoveryStateSnapshot.normal(), safety_override=False),
    )

    assert proposal.source == "manual"
    assert proposal.action_command is manual
    assert resolved.action_command is manual
    assert tuple(round(float(v), 4) for v in resolved.command_vector.tolist()) == (0.2, 0.0, 0.1)
    assert resolved.status is not None and resolved.status.state == "running"
    assert resolved.metadata["recovery_state"]["current_state"] == "NORMAL"


def test_command_resolver_zeroes_command_for_replan_pending() -> None:
    resolver = CommandResolver()
    proposal = resolver.resolve_action_command(
        manual_command=ActionCommand(action_type="NAV_TO_POSE", task_id="task"),
        task_command=None,
    )
    execution = LocomotionProposal(
        command_vector=np.asarray([0.4, 0.0, 0.0], dtype=np.float32),
        trajectory_update=TrajectoryUpdate(
            trajectory_world=np.asarray([[0.1, 0.0, 0.0]], dtype=np.float32),
            plan_version=2,
            stats=PlannerStats(last_plan_step=2),
            source_frame_id=2,
        ),
        evaluation=CommandEvaluation(force_stop=False, goal_distance_m=0.5, yaw_error_rad=0.0, reached_goal=False),
    )
    resolved = resolver.resolve_execution(
        proposal=proposal,
        execution=execution,
        recovery_state=RecoveryStateSnapshot(current_state=RecoveryState.REPLAN_PENDING.value, last_trigger_reason="trajectory_stale"),
        safety_decision=SafetyDecision(
            recovery_state=RecoveryStateSnapshot(current_state=RecoveryState.REPLAN_PENDING.value, last_trigger_reason="trajectory_stale"),
            safety_override=False,
        ),
    )
    assert tuple(float(v) for v in resolved.command_vector.tolist()) == (0.0, 0.0, 0.0)
    assert resolved.status is not None and resolved.status.state == "stale"


def test_command_resolver_fails_terminal_recovery_state() -> None:
    resolver = CommandResolver()
    proposal = resolver.resolve_action_command(
        manual_command=ActionCommand(action_type="NAV_TO_POSE", task_id="task"),
        task_command=None,
    )
    execution = LocomotionProposal(
        command_vector=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
        trajectory_update=TrajectoryUpdate(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=3,
            stats=PlannerStats(last_error="planner failed", last_plan_step=3),
            source_frame_id=3,
        ),
        evaluation=CommandEvaluation(force_stop=True, goal_distance_m=-1.0, yaw_error_rad=0.0, reached_goal=False),
    )
    failed_state = RecoveryStateSnapshot(current_state=RecoveryState.FAILED.value, last_trigger_reason="planner failed")
    resolved = resolver.resolve_execution(
        proposal=proposal,
        execution=execution,
        recovery_state=failed_state,
        safety_decision=SafetyDecision(recovery_state=failed_state, safety_override=True, safety_reason="planner failed"),
    )
    assert resolved.status is not None and resolved.status.state == "failed"
    assert resolved.safety_override is True
