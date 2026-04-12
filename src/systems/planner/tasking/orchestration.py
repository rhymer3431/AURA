from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from systems.planner.tasking.schemas import CompletionDecision
from systems.planner.tasking.validator import validate_subgoal, validate_task_frame_response


def _build_subgoal(
    subgoal_id: int,
    subgoal_type: str,
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"sg{subgoal_id}",
        "type": subgoal_type,
        "status": "pending",
        "succeed": False,
        "input": input_payload,
        "output": {},
        "attempts": 0,
        "failure_reason": None,
    }


class SubgoalExpander:
    def expand(self, task_frame: dict[str, Any]) -> list[dict[str, Any]]:
        task_frame = validate_task_frame_response(task_frame)
        subgoals: list[dict[str, Any]] = []
        target = task_frame["target"]
        query = task_frame["query"]
        constraints = task_frame["constraints"]
        clarification = task_frame["clarification"]
        base_target = {
            "object": target.get("object"),
            "instance_hint": target.get("instance_hint"),
            "location_hint": target.get("location_hint"),
        }

        def append_subgoal(subgoal_type: str, input_payload: dict[str, Any]) -> None:
            subgoals.append(_build_subgoal(len(subgoals) + 1, subgoal_type, input_payload))

        if task_frame["intent"] == "check_state":
            append_subgoal("navigate", {"target": base_target})
            append_subgoal("inspect", {"target": base_target, "query": query})
            if constraints["return_after_check"]:
                append_subgoal("return", {"target": {"object": "user"}})
            if constraints["report_result"]:
                append_subgoal(
                    "report",
                    {
                        "intent": task_frame["intent"],
                        "target": base_target,
                        "query": query,
                        "clarification": clarification,
                    },
                )
            return subgoals

        if task_frame["intent"] in {"find_object", "navigate_to_object"}:
            append_subgoal("navigate", {"target": base_target})
            if constraints["return_after_check"]:
                append_subgoal("return", {"target": {"object": "user"}})
            if constraints["report_result"]:
                append_subgoal(
                    "report",
                    {
                        "intent": task_frame["intent"],
                        "target": base_target,
                        "query": query,
                        "clarification": clarification,
                    },
                )
            return subgoals

        if constraints["report_result"]:
            append_subgoal(
                "report",
                {
                    "intent": task_frame["intent"],
                    "target": base_target,
                    "query": query,
                    "clarification": clarification,
                },
            )
        return subgoals


class NavigateExecutor(ABC):
    @abstractmethod
    def execute(self, subgoal: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError


class InspectExecutor(ABC):
    @abstractmethod
    def execute(self, subgoal: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError


class ReportExecutor(ABC):
    @abstractmethod
    def execute(self, subgoal: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError


class NavigateCompletionChecker:
    def __init__(self, stable_stop_frames: int = 3, distance_threshold: float | None = None) -> None:
        self.stable_stop_frames = stable_stop_frames
        self.distance_threshold = distance_threshold

    def check(
        self,
        subgoal: dict[str, Any],
        raw_output: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> CompletionDecision:
        del subgoal, runtime
        if raw_output.get("error"):
            return CompletionDecision(done=True, success=False, reason=str(raw_output["error"]))
        stop_requested = raw_output.get("last_vlm_output") == "STOP"
        target_visible = bool(raw_output.get("target_visible"))
        robot_stopped = bool(raw_output.get("robot_stopped"))
        stable_stop_count = int(raw_output.get("stable_stop_count", 0))
        distance_to_target = raw_output.get("distance_to_target")
        within_distance = True
        if self.distance_threshold is not None and isinstance(distance_to_target, (int, float)):
            within_distance = float(distance_to_target) <= self.distance_threshold
        if stop_requested and target_visible and robot_stopped and stable_stop_count >= self.stable_stop_frames and within_distance:
            return CompletionDecision(
                done=True,
                success=True,
                reason="stable_stop_with_target_visible",
            )
        return CompletionDecision(
            done=False,
            success=False,
            reason="navigation_incomplete",
            retryable=True,
        )


class InspectCompletionChecker:
    def __init__(self, stable_frames: int = 3, min_confidence: float = 0.8) -> None:
        self.stable_frames = stable_frames
        self.min_confidence = min_confidence

    def check(
        self,
        subgoal: dict[str, Any],
        raw_output: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> CompletionDecision:
        del subgoal, runtime
        if raw_output.get("error"):
            return CompletionDecision(done=True, success=False, reason=str(raw_output["error"]))
        observations = raw_output.get("observations")
        if isinstance(observations, list) and len(observations) >= self.stable_frames:
            tail = observations[-self.stable_frames :]
            consistent = len(set(tail)) == 1
        else:
            consistent = raw_output.get("observation_consistent", False)
        confidence = raw_output.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        observed_value = raw_output.get("observed_value")
        if (
            bool(raw_output.get("target_visible"))
            and observed_value is not None
            and consistent
            and float(confidence) >= self.min_confidence
        ):
            return CompletionDecision(done=True, success=True, reason="inspection_settled")
        return CompletionDecision(
            done=False,
            success=False,
            reason="inspection_incomplete",
            retryable=True,
        )


class ReportCompletionChecker:
    def check(
        self,
        subgoal: dict[str, Any],
        raw_output: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> CompletionDecision:
        del subgoal, runtime
        if raw_output.get("error"):
            return CompletionDecision(done=True, success=False, reason=str(raw_output["error"]))
        if raw_output.get("message") or raw_output.get("delivered"):
            return CompletionDecision(done=True, success=True, reason="report_generated")
        return CompletionDecision(
            done=False,
            success=False,
            reason="report_incomplete",
            retryable=True,
        )


class SubgoalStateMachine:
    def current_subgoal(self, subgoals: list[dict[str, Any]]) -> dict[str, Any] | None:
        for subgoal in subgoals:
            validate_subgoal(subgoal)
            if subgoal["status"] in {"pending", "running"}:
                return subgoal
        return None

    def begin(self, subgoal: dict[str, Any]) -> None:
        validate_subgoal(subgoal)
        if subgoal["status"] == "pending":
            subgoal["status"] = "running"

    def apply(
        self,
        subgoal: dict[str, Any],
        raw_output: dict[str, Any],
        decision: CompletionDecision,
    ) -> None:
        validate_subgoal(subgoal)
        subgoal["attempts"] += 1
        subgoal["output"] = raw_output
        if decision.done and decision.success:
            subgoal["status"] = "succeeded"
            subgoal["succeed"] = True
            subgoal["failure_reason"] = None
            return
        if decision.done and not decision.success:
            subgoal["status"] = "failed"
            subgoal["succeed"] = False
            subgoal["failure_reason"] = decision.reason
            return
        if decision.retryable:
            subgoal["status"] = "running"
            subgoal["succeed"] = False
            subgoal["failure_reason"] = None
            return
        subgoal["status"] = "failed"
        subgoal["succeed"] = False
        subgoal["failure_reason"] = decision.reason

    def is_complete(self, subgoals: list[dict[str, Any]]) -> bool:
        return all(subgoal["status"] in {"succeeded", "failed"} for subgoal in subgoals)


class SubgoalOrchestrator:
    def __init__(
        self,
        navigate_executor: NavigateExecutor,
        inspect_executor: InspectExecutor,
        report_executor: ReportExecutor,
        navigate_checker: NavigateCompletionChecker | None = None,
        inspect_checker: InspectCompletionChecker | None = None,
        report_checker: ReportCompletionChecker | None = None,
        expander: SubgoalExpander | None = None,
        state_machine: SubgoalStateMachine | None = None,
    ) -> None:
        self.navigate_executor = navigate_executor
        self.inspect_executor = inspect_executor
        self.report_executor = report_executor
        self.navigate_checker = navigate_checker or NavigateCompletionChecker()
        self.inspect_checker = inspect_checker or InspectCompletionChecker()
        self.report_checker = report_checker or ReportCompletionChecker()
        self.expander = expander or SubgoalExpander()
        self.state_machine = state_machine or SubgoalStateMachine()

    def initialize(self, task_frame: dict[str, Any]) -> list[dict[str, Any]]:
        return self.expander.expand(task_frame)

    def step(
        self,
        subgoals: list[dict[str, Any]],
        runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        subgoal = self.state_machine.current_subgoal(subgoals)
        if subgoal is None:
            return None
        self.state_machine.begin(subgoal)
        raw_output, decision = self._execute_and_check(subgoal, runtime or {})
        self.state_machine.apply(subgoal, raw_output, decision)
        return {
            "subgoal_id": subgoal["id"],
            "type": subgoal["type"],
            "raw_output": raw_output,
            "decision": {
                "done": decision.done,
                "success": decision.success,
                "reason": decision.reason,
                "retryable": decision.retryable,
            },
            "status": subgoal["status"],
        }

    def _execute_and_check(
        self,
        subgoal: dict[str, Any],
        runtime: dict[str, Any],
    ) -> tuple[dict[str, Any], CompletionDecision]:
        if subgoal["type"] in {"navigate", "return"}:
            raw_output = self.navigate_executor.execute(subgoal, runtime)
            return raw_output, self.navigate_checker.check(subgoal, raw_output, runtime)
        if subgoal["type"] == "inspect":
            raw_output = self.inspect_executor.execute(subgoal, runtime)
            return raw_output, self.inspect_checker.check(subgoal, raw_output, runtime)
        raw_output = self.report_executor.execute(subgoal, runtime)
        return raw_output, self.report_checker.check(subgoal, raw_output, runtime)
