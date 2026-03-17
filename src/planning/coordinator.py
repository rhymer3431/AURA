"""Planning coordinator aliases for System2/System1 backends."""

from __future__ import annotations

from services.dual_orchestrator import DualOrchestrator


class PlanningCoordinator(DualOrchestrator):
    """Compatibility-first planning alias for the dual planner coordinator."""
