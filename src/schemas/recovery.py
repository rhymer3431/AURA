from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


def _dict(payload: object) -> dict[str, Any]:
    return dict(payload) if isinstance(payload, dict) else {}


class RecoveryState(str, Enum):
    NORMAL = "NORMAL"
    REPLAN_PENDING = "REPLAN_PENDING"
    WAIT_SENSOR = "WAIT_SENSOR"
    SAFE_STOP = "SAFE_STOP"
    RECOVERY_TURN = "RECOVERY_TURN"
    FAILED = "FAILED"

    @classmethod
    def coerce(cls, value: object) -> RecoveryState:
        try:
            return cls(str(value))
        except ValueError:
            return cls.NORMAL


@dataclass(frozen=True)
class RecoveryStateSnapshot:
    current_state: str = RecoveryState.NORMAL.value
    entered_at_ns: int = 0
    retry_count: int = 0
    backoff_until_ns: int = 0
    last_trigger_reason: str = ""

    @property
    def state(self) -> RecoveryState:
        return RecoveryState.coerce(self.current_state)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def normal(cls) -> RecoveryStateSnapshot:
        return cls()

    @classmethod
    def from_dict(cls, payload: object) -> RecoveryStateSnapshot:
        data = _dict(payload)
        return cls(
            current_state=RecoveryState.coerce(data.get("current_state")).value,
            entered_at_ns=int(data.get("entered_at_ns", 0) or 0),
            retry_count=int(data.get("retry_count", 0) or 0),
            backoff_until_ns=int(data.get("backoff_until_ns", 0) or 0),
            last_trigger_reason=str(data.get("last_trigger_reason", "")),
        )


__all__ = ["RecoveryState", "RecoveryStateSnapshot"]
