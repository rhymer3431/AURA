from __future__ import annotations

from dataclasses import dataclass

from control.subgoal_planner import SubgoalPlanner
from memory.models import RecallResult, WorkingMemoryCandidate


@dataclass
class ActiveSearchPlan:
    recall_result: RecallResult
    candidate_index: int = 0

    @property
    def current_candidate(self) -> WorkingMemoryCandidate | None:
        if self.candidate_index < 0 or self.candidate_index >= len(self.recall_result.candidates):
            return None
        return self.recall_result.candidates[self.candidate_index]


class ObjectSearchService:
    def __init__(self, memory_service, subgoal_planner: SubgoalPlanner) -> None:  # noqa: ANN001
        self._memory = memory_service
        self._subgoal_planner = subgoal_planner
        self._active_plan: ActiveSearchPlan | None = None

    @property
    def active_plan(self) -> ActiveSearchPlan | None:
        return self._active_plan

    def begin_search(
        self,
        *,
        command_text: str,
        target_class: str,
        task_id: str,
        current_pose: tuple[float, float, float] | None = None,
        room_id: str = "",
    ):
        recall_result = self._memory.recall_object(
            query_text=command_text,
            target_class=target_class,
            intent="find",
            room_id=room_id,
            current_pose=current_pose,
        )
        self._active_plan = ActiveSearchPlan(recall_result=recall_result)
        return self._build_nav_command(task_id=task_id)

    def remaining_candidates(self) -> int:
        if self._active_plan is None:
            return 0
        active_count = sum(1 for candidate in self._active_plan.recall_result.candidates if candidate.active)
        return max(active_count - self._active_plan.candidate_index - 1, 0)

    def replan_next(self, *, task_id: str):
        if self._active_plan is None:
            return None
        self._active_plan.candidate_index += 1
        return self._build_nav_command(task_id=task_id)

    def current_navigation_command(self, *, task_id: str):
        return self._build_nav_command(task_id=task_id)

    def local_search(self, *, task_id: str):
        if self._active_plan is None:
            return self._subgoal_planner.local_search(task_id=task_id)
        current = self._active_plan.current_candidate
        if current is None:
            return self._subgoal_planner.local_search(task_id=task_id)
        place = self._memory.spatial_store.places.get(current.place_id)
        pose = place.pose if place is not None else None
        return self._subgoal_planner.local_search(
            task_id=task_id,
            place_id=current.place_id,
            target_pose_xyz=pose,
            metadata={"candidate_id": current.candidate_id},
        )

    def _build_nav_command(self, *, task_id: str):
        if self._active_plan is None:
            return None
        current = self._active_plan.current_candidate
        if current is None or not current.active:
            return None
        place = self._memory.spatial_store.places.get(current.place_id)
        if place is None:
            return None
        return self._subgoal_planner.nav_to_place(
            task_id=task_id,
            place_id=place.place_id,
            target_pose_xyz=place.pose,
            metadata={"candidate_id": current.candidate_id, "object_id": current.object_id},
        )

    def clear(self) -> None:
        self._active_plan = None
