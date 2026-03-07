from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BehaviorState(str, Enum):
    IDLE = "Idle"
    ATTEND_CALLER = "AttendCaller"
    FOLLOW_TARGET = "FollowTarget"
    GO_TO_REMEMBERED_OBJECT = "GoToRememberedObject"
    LOCAL_SEARCH = "LocalSearch"
    RECOVER_LOST_TARGET = "RecoverLostTarget"


@dataclass
class TransitionRecord:
    previous_state: BehaviorState
    next_state: BehaviorState
    reason: str


class BehaviorFSM:
    def __init__(self) -> None:
        self._state = BehaviorState.IDLE
        self._history: list[TransitionRecord] = []

    @property
    def state(self) -> BehaviorState:
        return self._state

    @property
    def history(self) -> list[TransitionRecord]:
        return list(self._history)

    def transition(self, next_state: BehaviorState, reason: str) -> TransitionRecord:
        record = TransitionRecord(previous_state=self._state, next_state=next_state, reason=str(reason))
        self._state = next_state
        self._history.append(record)
        return record
