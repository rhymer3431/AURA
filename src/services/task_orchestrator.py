from __future__ import annotations

from dataclasses import dataclass

from control.behavior_fsm import BehaviorFSM, BehaviorState
from control.critic import PlanCritic
from control.recovery import RecoveryPlanner
from control.subgoal_planner import SubgoalPlanner
from ipc.messages import ActionCommand, ActionStatus, TaskRequest
from perception.person_tracker import PersonTracker
from perception.speaker_events import SpeakerEvent

from .attention_service import AttentionService
from .follow_service import FollowService
from .intent_service import IntentService, ParsedIntent
from .memory_service import MemoryService
from .object_search_service import ObjectSearchService


@dataclass(frozen=True)
class TaskOrchestratorConfig:
    follow_reacquire_age_sec: float = 6.0


class TaskOrchestrator:
    def __init__(self, memory_service: MemoryService, *, config: TaskOrchestratorConfig | None = None) -> None:
        self.memory_service = memory_service
        self.config = config or TaskOrchestratorConfig()
        self.fsm = BehaviorFSM()
        self.intent_service = IntentService()
        self.subgoal_planner = SubgoalPlanner()
        self.attention_service = AttentionService()
        self.follow_service = FollowService(memory_service.temporal_store, PersonTracker(), self.subgoal_planner)
        self.object_search_service = ObjectSearchService(memory_service, self.subgoal_planner)
        self.critic = PlanCritic()
        self.recovery = RecoveryPlanner(memory_service.temporal_store, self.subgoal_planner)
        self._active_request: TaskRequest | None = None
        self._active_intent: ParsedIntent | None = None
        self._active_episode_id = ""

    @property
    def state(self) -> BehaviorState:
        return self.fsm.state

    def submit_task(self, request: TaskRequest) -> None:
        self._active_request = request
        self._active_intent = self.intent_service.parse(request.command_text, target_json=request.target_json)
        self._active_episode_id = self.memory_service.start_episode(
            command_text=request.command_text,
            intent=self._active_intent.name,
            target_json={"target_class": self._active_intent.target_class, **request.target_json},
        )
        if self._active_intent.name == "follow":
            track_id = self._active_intent.target_track_id or self.attention_service.bound_track_id or request.speaker_id
            self.follow_service.bind_target(track_id)
            self.fsm.transition(BehaviorState.FOLLOW_TARGET, "follow_request")
        elif self._active_intent.name == "goto_remembered_object":
            self.fsm.transition(BehaviorState.GO_TO_REMEMBERED_OBJECT, "memory_recall_request")
        elif self._active_intent.name == "attend_caller":
            self.fsm.transition(BehaviorState.ATTEND_CALLER, "attend_request")
        else:
            self.fsm.transition(BehaviorState.LOCAL_SEARCH, "default_local_search")

    def on_speaker_event(self, event: SpeakerEvent) -> None:
        self.memory_service.record_speaker_event(event)
        self.attention_service.record_event(event)
        if self.fsm.state == BehaviorState.IDLE:
            self.fsm.transition(BehaviorState.ATTEND_CALLER, "speaker_event")

    def on_observations(self, observations) -> None:  # noqa: ANN001
        self.memory_service.observe_objects(list(observations))
        self.attention_service.bind_person(list(observations))
        self.follow_service.ingest_observations(list(observations))

    def step(
        self,
        *,
        now: float,
        robot_pose: tuple[float, float, float] | None = None,
        action_status: ActionStatus | None = None,
    ) -> ActionCommand | None:
        state = self.fsm.state
        request = self._active_request
        if state == BehaviorState.IDLE:
            return None

        if state == BehaviorState.ATTEND_CALLER:
            event = self.attention_service.current_event()
            if event is None:
                self.fsm.transition(BehaviorState.IDLE, "event_cleared")
                return None
            return self.subgoal_planner.look_at(event, task_id=request.task_id if request is not None else "")

        if state == BehaviorState.FOLLOW_TARGET:
            if action_status is not None and action_status.state == "failed":
                self.fsm.transition(BehaviorState.RECOVER_LOST_TARGET, action_status.reason or "follow_failed")
            else:
                command = self.follow_service.build_command(task_id=request.task_id if request is not None else "")
                if command is not None:
                    return command
                self.fsm.transition(BehaviorState.RECOVER_LOST_TARGET, "target_unavailable")

        if self.fsm.state == BehaviorState.RECOVER_LOST_TARGET:
            command = self.recovery.recover_follow_target(
                self.follow_service.target_track_id,
                now=now,
                task_id=request.task_id if request is not None else "",
            )
            if command is not None:
                return command
            self.memory_service.finish_episode(success=False, failure_reason="follow_target_lost")
            self.fsm.transition(BehaviorState.IDLE, "recovery_exhausted")
            return self.subgoal_planner.stop(task_id=request.task_id if request is not None else "", reason="recovery_exhausted")

        if self.fsm.state == BehaviorState.GO_TO_REMEMBERED_OBJECT:
            if self.object_search_service.active_plan is None and request is not None and self._active_intent is not None:
                return self.object_search_service.begin_search(
                    command_text=request.command_text,
                    target_class=self._active_intent.target_class,
                    task_id=request.task_id,
                    current_pose=robot_pose,
                    room_id=str(request.target_json.get("room_id", "")),
                )
            if action_status is not None and action_status.state == "succeeded":
                self.fsm.transition(BehaviorState.LOCAL_SEARCH, "arrived_at_candidate")
                return self.object_search_service.local_search(task_id=request.task_id if request is not None else "")
            if action_status is not None and action_status.state == "failed":
                feedback = self.critic.evaluate(
                    action_status,
                    remaining_candidates=self.object_search_service.remaining_candidates(),
                )
                if feedback.should_replan and request is not None:
                    return self.object_search_service.replan_next(task_id=request.task_id)
                self.memory_service.finish_episode(success=False, failure_reason=feedback.reason)
                self.fsm.transition(BehaviorState.IDLE, "search_exhausted")
                return self.subgoal_planner.stop(task_id=request.task_id if request is not None else "", reason=feedback.reason)
            if self.object_search_service.active_plan is not None and request is not None:
                return self.object_search_service.current_navigation_command(task_id=request.task_id)

        if self.fsm.state == BehaviorState.LOCAL_SEARCH:
            if request is None:
                return self.subgoal_planner.local_search()
            return self.object_search_service.local_search(task_id=request.task_id)

        return None

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.fsm.state.value,
            "active_task_id": self._active_request.task_id if self._active_request is not None else "",
            "active_intent": self._active_intent.name if self._active_intent is not None else "",
            "active_episode_id": self._active_episode_id,
            "follow_target_id": self.follow_service.target_track_id,
        }

    def cancel_active_task(self, *, reason: str = "cancelled") -> None:
        if self._active_episode_id != "":
            self.memory_service.finish_episode(success=False, failure_reason=reason)
        self._active_request = None
        self._active_intent = None
        self._active_episode_id = ""
        self.object_search_service.clear()
        self.follow_service.clear_target()
        self.fsm.transition(BehaviorState.IDLE, reason)
