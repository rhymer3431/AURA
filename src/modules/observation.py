"""Observation module wrappers for the canonical navigation runtime."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.planning_session import ExecutionObservation, PlanningSession


@dataclass
class ObservationModule:
    """Frame-capture facade kept separate from planner ownership."""

    planning_session: PlanningSession

    def capture(self, frame_id: int, *, env=None) -> ExecutionObservation | None:  # noqa: ANN001
        return self.planning_session.capture_observation(frame_id, env=env)
