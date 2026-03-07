from __future__ import annotations

from memory.models import EpisodeRecord, SemanticRule
from memory.semantic_store import SemanticMemoryStore


class SemanticConsolidationService:
    def __init__(self, semantic_store: SemanticMemoryStore) -> None:
        self._semantic = semantic_store

    def summarize_episode(self, episode: EpisodeRecord) -> EpisodeRecord:
        target_class = str(episode.target_json.get("target_class", "")).strip().lower()
        room_id = str(episode.target_json.get("room_id", "")).strip().lower()
        tags = {
            episode.intent,
            "success" if bool(episode.success) else "failure",
        }
        if target_class != "":
            tags.add(target_class)
        if room_id != "":
            tags.add(room_id)
        if episode.failure_reason != "":
            tags.add(str(episode.failure_reason).strip().lower())
        for action in episode.recovery_actions:
            normalized = str(action).strip().lower()
            if normalized != "":
                tags.add(normalized)
        episode.summary_tags = sorted(tag for tag in tags if tag != "")
        if episode.summary_text == "":
            outcome = "succeeded" if bool(episode.success) else f"failed:{episode.failure_reason or 'unknown'}"
            candidates = ",".join(episode.candidate_place_ids[:3]) or "none"
            episode.summary_text = (
                f"{episode.intent}:{target_class or 'unknown'}:{room_id or 'any'} "
                f"{outcome} candidates={candidates} recovery={','.join(episode.recovery_actions) or 'none'}"
            )
        return episode

    def consolidate(self, episode: EpisodeRecord) -> list[SemanticRule]:
        self.summarize_episode(episode)
        generated: list[SemanticRule] = []
        target_class = str(episode.target_json.get("target_class", "")).strip().lower()
        room_id = str(episode.target_json.get("room_id", "")).strip().lower()

        if episode.intent == "goto_remembered_object" and target_class != "":
            rule_key = ":".join(token for token in ("find", target_class, room_id) if token != "")
            preferred_place_id = episode.candidate_place_ids[0] if episode.success and episode.candidate_place_ids else ""
            preferred_object_id = episode.candidate_object_ids[0] if episode.success and episode.candidate_object_ids else ""
            preferred_support = list(
                {
                    str(tag.split("surface:", maxsplit=1)[1]).strip().lower()
                    for tag in episode.summary_tags
                    if str(tag).startswith("surface:")
                }
            )
            generated.append(
                self._semantic.remember_rule(
                    rule_key,
                    f"find:{target_class}:{room_id or 'any'}",
                    succeeded=bool(episode.success),
                    trigger_signature=f"find:{target_class}:{room_id or 'any'}",
                    rule_type="object_search",
                    planner_hint={
                        "preferred_room": room_id,
                        "preferred_place_id": preferred_place_id,
                        "preferred_object_id": preferred_object_id,
                        "preferred_support_classes": preferred_support,
                    },
                    metadata={
                        "source_episode_id": episode.episode_id,
                        "summary_tags": list(episode.summary_tags),
                    },
                    now=episode.ended_at or episode.started_at,
                )
            )

        if episode.intent == "follow":
            corner_loss = (
                "corner_loss" in str(episode.failure_reason).lower()
                or any("corner" in str(action).lower() for action in episode.recovery_actions)
            )
            if corner_loss or episode.failure_reason == "follow_target_lost":
                generated.append(
                    self._semantic.remember_rule(
                        "follow:person:corner_loss",
                        "follow:person:corner_loss",
                        succeeded=bool(episode.success),
                        trigger_signature="follow:person:corner_loss",
                        rule_type="follow_recovery",
                        planner_hint={
                            "recovery_mode": "last_visible_corner_then_cone_search",
                            "follow_target_id": episode.follow_target_id,
                        },
                        metadata={
                            "source_episode_id": episode.episode_id,
                            "summary_tags": list(episode.summary_tags),
                        },
                        now=episode.ended_at or episode.started_at,
                    )
                )
        return generated
