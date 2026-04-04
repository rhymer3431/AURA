from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "AsyncNoGoalPlanner": ("control.async_planners", "AsyncNoGoalPlanner"),
    "AsyncPointGoalPlanner": ("control.async_planners", "AsyncPointGoalPlanner"),
    "BehaviorFSM": ("control.behavior_fsm", "BehaviorFSM"),
    "BehaviorState": ("control.behavior_fsm", "BehaviorState"),
    "CriticFeedback": ("control.critic", "CriticFeedback"),
    "NoGoalPlannerInput": ("control.async_planners", "NoGoalPlannerInput"),
    "PlanCritic": ("control.critic", "PlanCritic"),
    "PlannerInput": ("control.async_planners", "PlannerInput"),
    "PlannerOutput": ("control.async_planners", "PlannerOutput"),
    "RecoveryPlanner": ("control.recovery", "RecoveryPlanner"),
    "SubgoalPlanner": ("control.subgoal_planner", "SubgoalPlanner"),
    "TrackerResult": ("control.trajectory_tracker", "TrackerResult"),
    "TrajectoryTracker": ("control.trajectory_tracker", "TrajectoryTracker"),
    "TrajectoryTrackerConfig": ("control.trajectory_tracker", "TrajectoryTrackerConfig"),
    "TransitionRecord": ("control.behavior_fsm", "TransitionRecord"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
