from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schemas.recovery import RecoveryState, RecoveryStateSnapshot
from server.recovery_state import RecoveryEvent, RecoveryPolicy, apply_recovery_event


def _policy() -> RecoveryPolicy:
    return RecoveryPolicy(
        max_stale_age_ms=4000,
        planner_retry_budget=1,
        safe_stop_timeout_ms=20000,
        sensor_wait_budget_ms=100,
        recovery_turn_retry_limit=1,
        s2_retry_backoff_ms=50,
    )


def test_sensor_missing_waits_then_safe_stops() -> None:
    policy = _policy()
    initial = RecoveryStateSnapshot.normal()
    waiting = apply_recovery_event(initial, RecoveryEvent(kind="sensor_missing", reason="sensor_missing"), now_ns=10, policy=policy)
    assert waiting.state == RecoveryState.WAIT_SENSOR

    safe_stop = apply_recovery_event(waiting, RecoveryEvent(kind="sensor_missing", reason="sensor_missing"), now_ns=200_000_000, policy=policy)
    assert safe_stop.state == RecoveryState.SAFE_STOP
    assert safe_stop.last_trigger_reason == "sensor_missing"


def test_trajectory_stale_enters_replan_pending() -> None:
    next_state = apply_recovery_event(
        RecoveryStateSnapshot.normal(),
        RecoveryEvent(kind="trajectory_stale", reason="trajectory_stale"),
        now_ns=42,
        policy=_policy(),
    )
    assert next_state.state == RecoveryState.REPLAN_PENDING


def test_planning_failure_promotes_recovery_turn_then_failed() -> None:
    policy = _policy()
    first = apply_recovery_event(
        RecoveryStateSnapshot.normal(),
        RecoveryEvent(kind="planning_failure", reason="nav_failed"),
        now_ns=10,
        policy=policy,
    )
    assert first.state == RecoveryState.REPLAN_PENDING
    assert first.retry_count == 1

    second = apply_recovery_event(first, RecoveryEvent(kind="planning_failure", reason="nav_failed"), now_ns=20, policy=policy)
    assert second.state == RecoveryState.RECOVERY_TURN
    assert second.retry_count == 0

    third = apply_recovery_event(second, RecoveryEvent(kind="planning_failure", reason="nav_failed"), now_ns=30, policy=policy)
    assert third.state == RecoveryState.RECOVERY_TURN
    assert third.retry_count == 1

    failed = apply_recovery_event(third, RecoveryEvent(kind="planning_failure", reason="nav_failed"), now_ns=40, policy=policy)
    assert failed.state == RecoveryState.FAILED


def test_planning_success_and_task_reset_return_normal() -> None:
    policy = _policy()
    replan = apply_recovery_event(
        RecoveryStateSnapshot(current_state=RecoveryState.REPLAN_PENDING.value, entered_at_ns=5, retry_count=1, backoff_until_ns=15),
        RecoveryEvent(kind="planning_success", reason="planning_success"),
        now_ns=20,
        policy=policy,
    )
    assert replan.state == RecoveryState.NORMAL
    assert replan.retry_count == 0

    reset = apply_recovery_event(
        RecoveryStateSnapshot(current_state=RecoveryState.FAILED.value, entered_at_ns=50, retry_count=2, last_trigger_reason="nav_failed"),
        RecoveryEvent(kind="task_reset", reason="task_reset"),
        now_ns=100,
        policy=policy,
    )
    assert reset.state == RecoveryState.NORMAL
    assert reset.last_trigger_reason == "task_reset"
