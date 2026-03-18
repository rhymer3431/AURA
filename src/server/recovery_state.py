from __future__ import annotations

from dataclasses import dataclass

from schemas.recovery import RecoveryState, RecoveryStateSnapshot


@dataclass(frozen=True)
class RecoveryPolicy:
    max_stale_age_ms: int
    planner_retry_budget: int
    safe_stop_timeout_ms: int
    sensor_wait_budget_ms: int
    recovery_turn_retry_limit: int
    s2_retry_backoff_ms: int

    @classmethod
    def from_args(cls, args) -> RecoveryPolicy:  # noqa: ANN001
        return cls(
            max_stale_age_ms=int(max(float(getattr(args, "traj_max_stale_sec", 4.0)) * 1000.0, 0.0)),
            planner_retry_budget=max(int(getattr(args, "retry", 1)), 0),
            safe_stop_timeout_ms=int(max(float(getattr(args, "safety_timeout_sec", 20.0)) * 1000.0, 0.0)),
            sensor_wait_budget_ms=max(int(getattr(args, "sensor_wait_budget_ms", 0)), 0),
            recovery_turn_retry_limit=max(int(getattr(args, "recovery_turn_retry_limit", 1)), 0),
            s2_retry_backoff_ms=max(
                int(getattr(args, "s2_retry_backoff_ms", int(float(getattr(args, "s2_period_sec", 1.0)) * 1000.0))),
                0,
            ),
        )


@dataclass(frozen=True)
class RecoveryEvent:
    kind: str
    reason: str = ""


def is_terminal_recovery_state(state: RecoveryState | str) -> bool:
    current = RecoveryState.coerce(state)
    return current in {RecoveryState.SAFE_STOP, RecoveryState.FAILED}


def apply_recovery_event(
    current: RecoveryStateSnapshot,
    event: RecoveryEvent | None,
    *,
    now_ns: int,
    policy: RecoveryPolicy,
) -> RecoveryStateSnapshot:
    if event is None:
        return current

    state = current.state
    reason = str(event.reason or "")
    if event.kind == "task_reset":
        return RecoveryStateSnapshot(
            current_state=RecoveryState.NORMAL.value,
            entered_at_ns=int(now_ns),
            retry_count=0,
            backoff_until_ns=0,
            last_trigger_reason=reason,
        )

    if event.kind == "planning_success":
        return RecoveryStateSnapshot(
            current_state=RecoveryState.NORMAL.value,
            entered_at_ns=int(now_ns),
            retry_count=0,
            backoff_until_ns=0,
            last_trigger_reason=reason,
        )

    if event.kind == "sensor_restored":
        if state == RecoveryState.WAIT_SENSOR:
            return RecoveryStateSnapshot(
                current_state=RecoveryState.NORMAL.value,
                entered_at_ns=int(now_ns),
                retry_count=0,
                backoff_until_ns=0,
                last_trigger_reason=reason,
            )
        return current

    if event.kind == "timeout":
        if is_terminal_recovery_state(state):
            return current
        return RecoveryStateSnapshot(
            current_state=RecoveryState.SAFE_STOP.value,
            entered_at_ns=int(now_ns),
            retry_count=int(current.retry_count),
            backoff_until_ns=0,
            last_trigger_reason=reason or "timeout",
        )

    if event.kind == "sensor_missing":
        if state == RecoveryState.WAIT_SENSOR:
            budget_ns = int(max(policy.sensor_wait_budget_ms, 0)) * 1_000_000
            if budget_ns <= 0 or (current.entered_at_ns > 0 and int(now_ns) - int(current.entered_at_ns) >= budget_ns):
                return RecoveryStateSnapshot(
                    current_state=RecoveryState.SAFE_STOP.value,
                    entered_at_ns=int(now_ns),
                    retry_count=int(current.retry_count),
                    backoff_until_ns=0,
                    last_trigger_reason="sensor_missing",
                )
            return current
        if is_terminal_recovery_state(state):
            return current
        if int(policy.sensor_wait_budget_ms) > 0:
            return RecoveryStateSnapshot(
                current_state=RecoveryState.WAIT_SENSOR.value,
                entered_at_ns=int(now_ns),
                retry_count=0,
                backoff_until_ns=0,
                last_trigger_reason="sensor_missing",
            )
        return RecoveryStateSnapshot(
            current_state=RecoveryState.SAFE_STOP.value,
            entered_at_ns=int(now_ns),
            retry_count=0,
            backoff_until_ns=0,
            last_trigger_reason="sensor_missing",
        )

    if event.kind == "trajectory_stale":
        if is_terminal_recovery_state(state):
            return current
        return RecoveryStateSnapshot(
            current_state=RecoveryState.REPLAN_PENDING.value,
            entered_at_ns=int(now_ns),
            retry_count=0,
            backoff_until_ns=0,
            last_trigger_reason=reason or "trajectory_stale",
        )

    if event.kind == "planning_failure":
        if is_terminal_recovery_state(state) or state == RecoveryState.WAIT_SENSOR:
            return current
        if state == RecoveryState.RECOVERY_TURN:
            retry_count = int(current.retry_count) + 1
            next_state = RecoveryState.RECOVERY_TURN
            if retry_count > int(policy.recovery_turn_retry_limit):
                next_state = RecoveryState.FAILED
            return RecoveryStateSnapshot(
                current_state=next_state.value,
                entered_at_ns=int(now_ns),
                retry_count=0 if next_state == RecoveryState.FAILED else retry_count,
                backoff_until_ns=0 if next_state == RecoveryState.FAILED else int(now_ns) + int(policy.s2_retry_backoff_ms) * 1_000_000,
                last_trigger_reason=reason or "planning_failure",
            )
        retry_count = int(current.retry_count) + 1
        next_state = RecoveryState.REPLAN_PENDING
        next_retry = retry_count
        if retry_count > int(policy.planner_retry_budget):
            next_state = RecoveryState.RECOVERY_TURN
            next_retry = 0
        return RecoveryStateSnapshot(
            current_state=next_state.value,
            entered_at_ns=int(now_ns),
            retry_count=next_retry,
            backoff_until_ns=int(now_ns) + int(policy.s2_retry_backoff_ms) * 1_000_000,
            last_trigger_reason=reason or "planning_failure",
        )

    return current


__all__ = [
    "RecoveryEvent",
    "RecoveryPolicy",
    "apply_recovery_event",
    "is_terminal_recovery_state",
]
