from __future__ import annotations

from dataclasses import dataclass

from ipc.messages import ActionStatus


@dataclass(frozen=True)
class CriticFeedback:
    should_replan: bool
    reason: str
    next_candidate_index: int | None = None


class PlanCritic:
    def evaluate(
        self,
        status: ActionStatus | None,
        *,
        remaining_candidates: int,
        exhausted_reason: str = "candidate_exhausted",
    ) -> CriticFeedback:
        if status is None:
            return CriticFeedback(should_replan=False, reason="no_status")
        if status.state == "failed" and remaining_candidates > 0:
            return CriticFeedback(should_replan=True, reason=status.reason or "action_failed", next_candidate_index=0)
        if status.state == "failed":
            return CriticFeedback(should_replan=False, reason=exhausted_reason, next_candidate_index=None)
        return CriticFeedback(should_replan=False, reason="hold")
