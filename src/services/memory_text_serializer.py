from __future__ import annotations

from .memory_policy_types import MemoryPolicyContext, MemoryPolicyLabel


class MemoryTextSerializer:
    def serialize(self, context: MemoryPolicyContext) -> str:
        lines = [
            "You are a text-only memory policy controller for remembered-object navigation.",
            f"Instruction: {context.instruction}",
            f"Target class: {context.target_class or 'unknown'}",
            f"Task state: {context.task_state or 'unknown'}",
            f"Current pose: {self._format_pose(context.current_pose)}",
            f"Visible target now: {'yes' if context.visible_target_now else 'no'}",
            f"Candidate count: {int(context.candidate_count)}",
            f"Top score: {float(context.top_score):.4f}",
            f"Score gap: {float(context.score_gap):.4f}",
            f"Retrieval confidence: {float(context.retrieval_confidence):.4f}",
            f"Ambiguity: {'yes' if context.ambiguity else 'no'}",
            "Scratchpad:",
        ]
        scratchpad_lines = self._scratchpad_lines(context)
        if scratchpad_lines:
            lines.extend(f"- {line}" for line in scratchpad_lines)
        else:
            lines.append("- None")
        lines.append("Retrieved memory lines:")
        memory_lines = self._memory_lines(context)
        if memory_lines:
            lines.extend(f"- {line}" for line in memory_lines)
        else:
            lines.append("- None")
        lines.append("Semantic rule hints:")
        if context.semantic_rule_hints:
            lines.extend(f"- {line}" for line in context.semantic_rule_hints[:4])
        else:
            lines.append("- None")
        lines.append("Allowed labels:")
        lines.extend(f"- {label.value}" for label in MemoryPolicyLabel)
        lines.append("Return exactly one label.")
        return "\n".join(lines)

    @staticmethod
    def _format_pose(current_pose: tuple[float, float, float] | None) -> str:
        if current_pose is None:
            return "unknown"
        x, y, z = tuple(float(value) for value in current_pose[:3])
        return f"x={x:.3f}, y={y:.3f}, z={z:.3f}"

    @staticmethod
    def _scratchpad_lines(context: MemoryPolicyContext) -> list[str]:
        if context.memory_context is None or context.memory_context.scratchpad is None:
            return []
        scratchpad = context.memory_context.scratchpad
        lines: list[str] = []
        if scratchpad.goal_summary.strip() != "":
            lines.append(f"Goal: {scratchpad.goal_summary.strip()}")
        if scratchpad.checked_locations:
            lines.append("Checked: " + ", ".join(scratchpad.checked_locations[-3:]))
        if scratchpad.recent_hint.strip() != "":
            lines.append(f"Hint: {scratchpad.recent_hint.strip()}")
        if scratchpad.next_priority.strip() != "":
            lines.append(f"Next: {scratchpad.next_priority.strip()}")
        return lines[:4]

    @staticmethod
    def _memory_lines(context: MemoryPolicyContext) -> list[str]:
        if context.memory_context is None:
            return []
        return [str(line.text).strip() for line in context.memory_context.text_lines if str(line.text).strip() != ""][:5]
