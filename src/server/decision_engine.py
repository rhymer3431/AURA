from __future__ import annotations

from dataclasses import dataclass

from schemas.events import FrameEvent
from schemas.world_state import TaskSnapshot, WorldStateSnapshot


@dataclass(frozen=True)
class DecisionDirective:
    process_perception: bool
    retrieve_memory: bool
    use_manual_command: bool
    route_task_command: bool
    reason: str = ""


@dataclass(frozen=True)
class DualDecisionDirective:
    force_s2: bool
    launch_s2: bool
    launch_s1: bool
    goal_missing: bool
    goal_stale: bool
    traj_missing: bool
    traj_stale: bool
    discard_stale_traj: bool
    backoff_active: bool


class DecisionEngine:
    def evaluate(
        self,
        *,
        world_state: WorldStateSnapshot,
        task: TaskSnapshot,
        frame_event: FrameEvent,
        manual_command_present: bool,
        active_memory_instruction: str,
    ) -> DecisionDirective:
        process_perception = bool(frame_event.batch is not None and frame_event.observation is not None)
        retrieve_memory = process_perception and str(active_memory_instruction).strip() != ""
        route_task_command = not bool(manual_command_present)
        reason = "manual_command"
        if route_task_command:
            reason = "task_orchestrator"
        if not process_perception:
            reason = "sensor_unavailable"
        if retrieve_memory:
            reason = "memory_aware_planning"
        return DecisionDirective(
            process_perception=process_perception,
            retrieve_memory=retrieve_memory,
            use_manual_command=bool(manual_command_present),
            route_task_command=route_task_command,
            reason=reason,
        )

    def evaluate_dual(
        self,
        *,
        now: float,
        goal_cache,
        traj_cache,
        last_s1_ts: float,
        last_s2_ts: float,
        s1_period_sec: float,
        s2_period_sec: float,
        goal_ttl_sec: float,
        traj_ttl_sec: float,
        traj_max_stale_sec: float,
        s2_retry_after_ts: float,
        force_s2_pending: bool,
        events: dict[str, object] | None = None,
    ) -> DualDecisionDirective:
        goal = goal_cache
        traj = traj_cache
        event_map = dict(events or {})
        external_force = bool(event_map.get("force_s2", False)) or bool(event_map.get("stuck", False)) or bool(
            event_map.get("collision_risk", False)
        )
        force_s2 = bool(external_force or force_s2_pending)
        goal_missing = goal is None
        goal_stale = bool(goal is not None and (now - float(goal.updated_at)) > float(goal_ttl_sec))
        due_s2 = (now - float(last_s2_ts)) >= float(s2_period_sec)
        awaiting_first_traj = bool(goal is not None and traj is None and str(goal.mode) == "pixel_goal" and not bool(goal.stop))
        backoff_active = now < float(s2_retry_after_ts)
        should_s2 = bool(force_s2 or goal_missing or goal_stale or due_s2)
        if awaiting_first_traj and not force_s2:
            should_s2 = bool(goal_missing)
        if backoff_active and not force_s2:
            should_s2 = False

        goal_is_pixel = bool(
            goal is not None
            and str(goal.mode) == "pixel_goal"
            and getattr(goal, "pixel_x", None) is not None
            and getattr(goal, "pixel_y", None) is not None
            and not bool(goal.stop)
        )
        goal_changed = bool(goal_is_pixel and (traj is None or int(traj.goal_version) != int(goal.version)))
        traj_missing = traj is None
        traj_stale = bool(traj is not None and (now - float(traj.updated_at)) > float(traj_ttl_sec))
        discard_stale_traj = bool(traj is not None and (now - float(traj.updated_at)) > float(traj_max_stale_sec))
        due_s1 = (now - float(last_s1_ts)) >= float(s1_period_sec)
        should_s1 = bool(goal_is_pixel and (goal_changed or traj_missing or traj_stale or due_s1))
        return DualDecisionDirective(
            force_s2=force_s2,
            launch_s2=should_s2,
            launch_s1=should_s1,
            goal_missing=goal_missing,
            goal_stale=goal_stale,
            traj_missing=traj_missing,
            traj_stale=traj_stale,
            discard_stale_traj=discard_stale_traj,
            backoff_active=backoff_active,
        )
