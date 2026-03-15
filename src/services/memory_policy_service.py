from __future__ import annotations

from dataclasses import replace

from config.memory_policy_config import MemoryPolicyConfig

from .memory_policy_types import MemoryPolicyContext, MemoryPolicyDecision, MemoryPolicyLabel
from .memory_text_serializer import MemoryTextSerializer
from .memory_service import MemoryService
from .text_only_memory_controller import TextOnlyMemoryController


class MemoryPolicyService:
    def __init__(
        self,
        memory_service: MemoryService,
        *,
        config: MemoryPolicyConfig | None = None,
        serializer: MemoryTextSerializer | None = None,
        controller: TextOnlyMemoryController | None = None,
    ) -> None:
        self.memory_service = memory_service
        self.config = config or MemoryPolicyConfig()
        self.serializer = serializer or MemoryTextSerializer()
        self.controller = controller or TextOnlyMemoryController(self.config)

    def build_context(
        self,
        *,
        instruction: str,
        target_class: str,
        task_state: str,
        current_pose: tuple[float, float, float] | None,
        visible_target_now: bool,
        room_id: str = "",
    ) -> MemoryPolicyContext:
        memory_context = self.memory_service.build_memory_context(
            instruction=str(instruction),
            current_pose=current_pose,
        )
        recall_result = self.memory_service.preview_object_recall(
            query_text=str(instruction),
            target_class=str(target_class),
            intent="find",
            room_id=str(room_id),
            current_pose=current_pose,
        )
        active_candidates = [candidate for candidate in recall_result.candidates if candidate.active]
        top_score = float(active_candidates[0].score) if active_candidates else 0.0
        next_score = float(active_candidates[1].score) if len(active_candidates) > 1 else 0.0
        score_gap = max(top_score - next_score, 0.0)
        retrieval_confidence = min(max(top_score, 0.0), 1.0)
        ambiguity = len(active_candidates) > 1 and score_gap < float(self.config.ambiguity_gap_threshold)
        return MemoryPolicyContext(
            instruction=str(instruction).strip(),
            target_class=str(target_class).strip(),
            task_state=str(task_state).strip() or "active",
            current_pose=current_pose,
            visible_target_now=bool(visible_target_now),
            memory_context=memory_context,
            recall_result=recall_result,
            semantic_rule_hints=self._semantic_rule_hints(recall_result),
            candidate_count=len(active_candidates),
            top_score=top_score,
            score_gap=score_gap,
            retrieval_confidence=retrieval_confidence,
            ambiguity=ambiguity,
        )

    def evaluate(self, context: MemoryPolicyContext) -> tuple[MemoryPolicyDecision, str]:
        prompt_text = self.serializer.serialize(context)
        decision = self.controller.predict(prompt_text, context)
        shadow_only = self._shadow_only_for_label(decision.label)
        decision = replace(
            decision,
            shadow_only=bool(shadow_only),
            feature_snapshot={
                **context.feature_snapshot(),
                **dict(decision.feature_snapshot),
            },
        )
        return decision, prompt_text

    def evaluate_remembered_object(
        self,
        *,
        instruction: str,
        target_class: str,
        task_state: str,
        current_pose: tuple[float, float, float] | None,
        visible_target_now: bool,
        room_id: str = "",
    ) -> tuple[MemoryPolicyDecision, str, MemoryPolicyContext]:
        context = self.build_context(
            instruction=instruction,
            target_class=target_class,
            task_state=task_state,
            current_pose=current_pose,
            visible_target_now=visible_target_now,
            room_id=room_id,
        )
        decision, prompt_text = self.evaluate(context)
        return decision, prompt_text, context

    def _shadow_only_for_label(self, label: MemoryPolicyLabel) -> bool:
        if not bool(self.config.enabled):
            return True
        if bool(self.config.shadow_mode):
            return True
        if label in {MemoryPolicyLabel.ROUTE_DIRECT_VISION, MemoryPolicyLabel.WAIT}:
            return True
        if label == MemoryPolicyLabel.STOP and not bool(self.config.live_stop_enabled):
            return True
        if label in {MemoryPolicyLabel.TURN_LEFT, MemoryPolicyLabel.TURN_RIGHT} and not bool(self.config.live_turns_enabled):
            return True
        return False

    @staticmethod
    def _semantic_rule_hints(recall_result) -> list[str]:  # noqa: ANN001
        if recall_result is None:
            return []
        hints: list[str] = []
        for rule in recall_result.semantic_rules:
            planner_hint = dict(rule.planner_hint)
            parts = [str(rule.rule_key).strip() or str(rule.description).strip()]
            if planner_hint:
                hint_items = ", ".join(
                    f"{key}={planner_hint[key]}"
                    for key in sorted(planner_hint)
                    if str(planner_hint[key]).strip() != ""
                )
                if hint_items != "":
                    parts.append(hint_items)
            hint_text = " | ".join(part for part in parts if part != "")
            if hint_text != "":
                hints.append(hint_text)
        return hints[:4]
