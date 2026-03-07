from __future__ import annotations

import time

from .models import SemanticRule


class SemanticMemoryStore:
    def __init__(self) -> None:
        self._rules: dict[str, SemanticRule] = {}

    def remember_rule(
        self,
        rule_key: str,
        description: str,
        *,
        succeeded: bool,
        trigger_signature: str = "",
        rule_type: str = "heuristic",
        planner_hint: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        now: float | None = None,
    ) -> SemanticRule:
        key = str(rule_key)
        rule = self._rules.get(key)
        if rule is None:
            rule = SemanticRule(
                rule_key=key,
                description=str(description),
                trigger_signature=str(trigger_signature),
                rule_type=str(rule_type),
            )
            self._rules[key] = rule
        if trigger_signature != "":
            rule.trigger_signature = str(trigger_signature)
        if rule_type != "":
            rule.rule_type = str(rule_type)
        rule.support_count += 1
        if succeeded:
            rule.success_count += 1
        rule.success_rate = rule.success_count / max(rule.support_count, 1)
        rule.last_updated = float(now if now is not None else time.time())
        if planner_hint:
            rule.planner_hint.update(dict(planner_hint))
        if metadata:
            rule.metadata.update(metadata)
        return rule

    def matching_rules(self, *, intent: str, target_class: str, room_id: str = "") -> list[SemanticRule]:
        tokens = [token for token in (intent.strip().lower(), target_class.strip().lower(), room_id.strip().lower()) if token != ""]
        results = []
        for rule in self._rules.values():
            lowered = rule.rule_key.lower()
            trigger = rule.trigger_signature.lower()
            if all(token in lowered or token in trigger for token in tokens):
                results.append(rule)
        return sorted(results, key=lambda item: (item.success_rate, item.support_count), reverse=True)

    def list(self) -> list[SemanticRule]:
        return sorted(self._rules.values(), key=lambda item: (item.success_rate, item.support_count), reverse=True)
