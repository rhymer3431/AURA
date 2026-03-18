from __future__ import annotations

from dataclasses import dataclass

from schemas.events import FrameEvent
from schemas.recovery import RecoveryStateSnapshot

from .recovery_state import RecoveryEvent, RecoveryPolicy, apply_recovery_event


@dataclass(frozen=True)
class SafetyDecision:
    recovery_state: RecoveryStateSnapshot
    safety_override: bool
    safety_reason: str = ""


class SafetySupervisor:
    def __init__(self, args) -> None:
        self._policy = RecoveryPolicy.from_args(args)

    @property
    def policy(self) -> RecoveryPolicy:
        return self._policy

    def evaluate(
        self,
        *,
        frame_event: FrameEvent,
        trajectory_update,
        recovery_state: RecoveryStateSnapshot,
        now_ns: int,
    ) -> SafetyDecision:  # noqa: ANN001
        current = recovery_state
        frame_age_ms = max((float(frame_event.metadata.timestamp_ns) - float(frame_event.timestamp_ns)) / 1.0e6, 0.0)
        if frame_age_ms > float(self._policy.safe_stop_timeout_ms) > 0.0:
            next_state = apply_recovery_event(
                current,
                RecoveryEvent(kind="timeout", reason="timeout"),
                now_ns=now_ns,
                policy=self._policy,
            )
            return SafetyDecision(recovery_state=next_state, safety_override=True, safety_reason="timeout")

        if frame_event.observation is None:
            next_state = apply_recovery_event(
                current,
                RecoveryEvent(kind="sensor_missing", reason="sensor_missing"),
                now_ns=now_ns,
                policy=self._policy,
            )
            return SafetyDecision(
                recovery_state=next_state,
                safety_override=next_state.current_state != "NORMAL",
                safety_reason="sensor_missing" if next_state.current_state != "NORMAL" else "",
            )

        if current.current_state == "WAIT_SENSOR":
            next_state = apply_recovery_event(
                current,
                RecoveryEvent(kind="sensor_restored", reason="sensor_restored"),
                now_ns=now_ns,
                policy=self._policy,
            )
            if next_state != current:
                return SafetyDecision(recovery_state=next_state, safety_override=False, safety_reason="")

        stale_ms = max(float(getattr(trajectory_update, "stale_sec", -1.0)), 0.0) * 1000.0
        if stale_ms > float(self._policy.max_stale_age_ms) > 0.0:
            next_state = apply_recovery_event(
                current,
                RecoveryEvent(kind="trajectory_stale", reason="trajectory_stale"),
                now_ns=now_ns,
                policy=self._policy,
            )
            return SafetyDecision(recovery_state=next_state, safety_override=False, safety_reason="trajectory_stale")

        return SafetyDecision(recovery_state=current, safety_override=False, safety_reason="")
