from __future__ import annotations

from .models import SemanticRule


class SemanticMemoryStore:
    def __init__(self) -> None:
        self._rules: dict[str, SemanticRule] = {}

    def remember_rule(self, rule_key: str, description: str, *, succeeded: bool, metadata: dict[str, object] | None = None) -> None:
        key = str(rule_key)
        rule = self._rules.get(key)
        if rule is None:
            rule = SemanticRule(rule_key=key, description=str(description))
            self._rules[key] = rule
        rule.support_count += 1
        if succeeded:
            rule.success_count += 1
        rule.success_rate = rule.success_count / max(rule.support_count, 1)
        if metadata:
            rule.metadata.update(metadata)

    def matching_rules(self, *, intent: str, target_class: str, room_id: str = "") -> list[SemanticRule]:
        tokens = [token for token in (intent.strip().lower(), target_class.strip().lower(), room_id.strip().lower()) if token != ""]
        results = []
        for rule in self._rules.values():
            lowered = rule.rule_key.lower()
            if all(token in lowered for token in tokens):
                results.append(rule)
        return sorted(results, key=lambda item: (item.success_rate, item.support_count), reverse=True)

    def list(self) -> list[SemanticRule]:
        return sorted(self._rules.values(), key=lambda item: (item.success_rate, item.support_count), reverse=True)
