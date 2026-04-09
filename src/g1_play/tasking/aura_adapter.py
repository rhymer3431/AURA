from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .llm_client import CompletionFn
from .orchestration import (
    InspectCompletionChecker,
    InspectExecutor,
    NavigateCompletionChecker,
    NavigateExecutor,
    ReportCompletionChecker,
    ReportExecutor,
    SubgoalOrchestrator,
    SubgoalExpander,
)
from .planner_service import PlannerService
from .reporting import (
    build_inspection_question,
    build_navigation_instruction,
    observed_value_from_answer,
    render_report_message,
)
from .validator import validate_task_frame_response


DEFAULT_CAPABILITIES = {
    "detectable_objects": [
        "tv",
        "sofa",
        "bed",
        "chair",
        "refrigerator",
        "door",
        "purple_box_cart",
    ],
    "inspectable_attributes": {
        "tv": ["power_state"],
        "door": ["open_state"],
    },
    "can_return_home": True,
}


def build_plan_request(instruction: str) -> dict[str, Any]:
    return {
        "utterance_ko": str(instruction),
        "robot_state": {
            "current_room": None,
            "holding_object": None,
        },
        "world_summary": {
            "known_rooms": [],
            "recent_seen": [],
        },
        "capabilities": DEFAULT_CAPABILITIES,
    }


class AuraSubgoalExpander(SubgoalExpander):
    def expand(self, task_frame: dict[str, Any]) -> list[dict[str, Any]]:
        task_frame = validate_task_frame_response(task_frame, DEFAULT_CAPABILITIES)
        return super().expand(task_frame)


class AuraNavigateExecutor(NavigateExecutor):
    def execute(self, subgoal: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime = runtime or {}
        controller = runtime["controller"]
        task_state = runtime.get("task_state")
        previous = subgoal.get("output", {})

        if subgoal["type"] == "return":
            origin_pose = None if task_state is None else task_state.origin_pose
            if origin_pose is None:
                return {"error": "origin_pose_unavailable"}
            if not previous.get("started"):
                start_info = controller.start_return_to_origin(origin_pose)
            else:
                start_info = {}
            snapshot = controller.navigation_snapshot(origin_pose=origin_pose)
            return {"started": True, "mode": "return", "snapshot": snapshot, **start_info}

        prompt = build_navigation_instruction(subgoal["input"]["target"], language="en")
        if not previous.get("started") or previous.get("prompt") != prompt:
            start_info = controller.start_navigation_instruction(prompt, "en")
        else:
            start_info = {}
        origin_pose = None if task_state is None else task_state.origin_pose
        snapshot = controller.navigation_snapshot(origin_pose=origin_pose)
        return {"started": True, "mode": "navigate", "prompt": prompt, "snapshot": snapshot, **start_info}


class AuraInspectExecutor(InspectExecutor):
    def execute(self, subgoal: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime = runtime or {}
        controller = runtime["controller"]
        previous = subgoal.get("output", {})
        question = build_inspection_question(subgoal["input"]["target"], subgoal["input"]["query"])
        answer_text = controller.check_binary_question(question)
        if answer_text not in {"true", "false"}:
            return {"error": f"invalid_binary_answer:{answer_text}"}

        answer_is_true = answer_text == "true"
        observations = list(previous.get("observations", []))
        observations.append(answer_is_true)
        confidence = 0.0
        if observations:
            dominant = max(observations.count(True), observations.count(False))
            confidence = float(dominant / len(observations))
        observed_value = observed_value_from_answer(subgoal["input"]["query"], answer_is_true)
        return {
            "question": question,
            "answer": answer_text,
            "target_visible": True,
            "observations": observations,
            "observation_consistent": len(observations) >= 3 and len(set(observations[-3:])) == 1,
            "confidence": confidence,
            "observed_value": observed_value,
            "decision": bool(answer_is_true),
        }


class AuraReportExecutor(ReportExecutor):
    def execute(self, subgoal: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime = runtime or {}
        controller = runtime["controller"]
        task_state = runtime["task_state"]
        message = render_report_message(task_state.task_frame, task_state.subgoals)
        controller.set_last_report(message)
        return {"message": message, "delivered": True}


class AuraNavigateCompletionChecker(NavigateCompletionChecker):
    def check(
        self,
        subgoal: dict[str, Any],
        raw_output: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ):
        del runtime
        if raw_output.get("error"):
            return super().check(subgoal, raw_output, None)
        snapshot = raw_output.get("snapshot", {})
        locomotion = np.asarray(snapshot.get("locomotion_command", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(3)
        zero_locomotion = float(np.linalg.norm(locomotion)) <= 0.05
        stable_state = str(snapshot.get("state_label") or "") in {"done", "waiting", "tracking", "stale-hold"}

        if subgoal["type"] == "return":
            if bool(snapshot.get("return_pose_reached")) and zero_locomotion and stable_state:
                from .schemas import CompletionDecision

                return CompletionDecision(done=True, success=True, reason="return_goal_reached")
            from .schemas import CompletionDecision

            return CompletionDecision(done=False, success=False, reason="return_incomplete", retryable=True)

        system2_status = str(snapshot.get("system2_status") or "").strip().lower()
        system2_mode = str(snapshot.get("system2_decision_mode") or "").strip().lower()
        action_override_mode = snapshot.get("action_override_mode")
        planner_target_mode = str(snapshot.get("planner_target_mode") or "none")
        stop_like = system2_status == "stop" or system2_mode in {"stop", "wait"}
        if stop_like and action_override_mode is None and zero_locomotion and stable_state and planner_target_mode == "none":
            from .schemas import CompletionDecision

            return CompletionDecision(done=True, success=True, reason="navigate_stopped")
        from .schemas import CompletionDecision

        return CompletionDecision(done=False, success=False, reason="navigate_incomplete", retryable=True)


class AuraInspectCompletionChecker(InspectCompletionChecker):
    def check(
        self,
        subgoal: dict[str, Any],
        raw_output: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ):
        del subgoal, runtime
        from .schemas import CompletionDecision

        if raw_output.get("error"):
            return CompletionDecision(done=True, success=False, reason=str(raw_output["error"]))
        observations = list(raw_output.get("observations", []))
        if len(observations) < self.stable_frames:
            return CompletionDecision(done=False, success=False, reason="inspection_incomplete", retryable=True)
        tail = observations[-self.stable_frames :]
        consistent = len(set(tail)) == 1
        confidence = float(raw_output.get("confidence", 0.0))
        if consistent and confidence >= self.min_confidence:
            return CompletionDecision(done=True, success=True, reason="inspection_settled")
        return CompletionDecision(done=False, success=False, reason="inspection_incomplete", retryable=True)


class AuraTaskingAdapter:
    def __init__(
        self,
        completion: CompletionFn | None = None,
        *,
        model: str,
        timeout: float,
    ) -> None:
        self.planner_service = PlannerService(
            completion=completion,
            model=model,
            timeout=timeout,
        )
        self.orchestrator = SubgoalOrchestrator(
            navigate_executor=AuraNavigateExecutor(),
            inspect_executor=AuraInspectExecutor(),
            report_executor=AuraReportExecutor(),
            navigate_checker=AuraNavigateCompletionChecker(),
            inspect_checker=AuraInspectCompletionChecker(),
            report_checker=ReportCompletionChecker(),
            expander=AuraSubgoalExpander(),
        )

    def plan_task_frame(self, instruction: str) -> dict[str, Any]:
        return self.planner_service.plan_task_frame(build_plan_request(instruction))

    def initialize_subgoals(self, task_frame: dict[str, Any]) -> list[dict[str, Any]]:
        return self.orchestrator.initialize(task_frame)

    def step(self, subgoals: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any] | None:
        return self.orchestrator.step(subgoals, runtime)


@dataclass(slots=True)
class PlannerConfig:
    base_url: str
    model: str
    timeout: float
