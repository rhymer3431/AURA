from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.memory_policy_config import MemoryPolicyConfig
from memory.models import MemoryContextBundle, RetrievedMemoryLine, ScratchpadState
from memory.models import ObsObject
from services.memory_policy_service import MemoryPolicyService
from services.memory_policy_types import MemoryPolicyContext, MemoryPolicyLabel
from services.memory_text_serializer import MemoryTextSerializer
from services.memory_service import MemoryService
from services.text_only_memory_controller import TextOnlyMemoryController


def test_memory_text_serializer_renders_deterministic_prompt() -> None:
    serializer = MemoryTextSerializer()
    context = MemoryPolicyContext(
        instruction="아까 봤던 사과를 찾아가",
        target_class="apple",
        task_state="active",
        current_pose=(1.2, 0.4, 0.0),
        visible_target_now=False,
        memory_context=MemoryContextBundle(
            instruction="아까 봤던 사과를 찾아가",
            scratchpad=ScratchpadState(
                instruction="아까 봤던 사과를 찾아가",
                planner_mode="interactive",
                task_state="active",
                goal_summary="Find apple.",
                checked_locations=["hallway"],
                recent_hint="Observed apple in kitchen.",
                next_priority="Use the remembered evidence.",
            ),
            text_lines=[
                RetrievedMemoryLine(
                    text="apple seen in kitchen on the left.",
                    score=0.8,
                    source_type="object_memory",
                    entity_id="apple_1",
                )
            ],
        ),
        recall_result=None,
        semantic_rule_hints=["find:apple:kitchen | preferred_room=kitchen"],
        candidate_count=2,
        top_score=0.82,
        score_gap=0.08,
        retrieval_confidence=0.82,
        ambiguity=True,
    )

    prompt_text = serializer.serialize(context)

    assert "Instruction: 아까 봤던 사과를 찾아가" in prompt_text
    assert "apple seen in kitchen on the left." in prompt_text
    assert "preferred_room=kitchen" in prompt_text
    assert prompt_text.endswith("Return exactly one label.")


def test_text_only_memory_controller_falls_back_after_parse_failure(monkeypatch) -> None:  # noqa: ANN001
    controller = TextOnlyMemoryController(
        MemoryPolicyConfig(backend="hf_generate", model_name_or_path="dummy-model")
    )
    context = MemoryPolicyContext(
        instruction="아까 봤던 사과를 찾아가",
        target_class="apple",
        task_state="active",
        current_pose=(0.0, 0.0, 0.0),
        visible_target_now=False,
        memory_context=None,
        recall_result=None,
        candidate_count=1,
        top_score=0.74,
        score_gap=0.3,
        retrieval_confidence=0.74,
        ambiguity=False,
    )
    monkeypatch.setattr(controller, "_generate_label", lambda prompt_text: "TURN_LEFT now")

    decision = controller.predict("prompt", context)

    assert decision.label == MemoryPolicyLabel.WAIT
    assert decision.source == "hf_generate"
    assert decision.fallback_used is True
    assert decision.feature_snapshot["error_kind"] == "parse_failure"


def test_memory_policy_service_derives_retrieval_features() -> None:
    memory_service = MemoryService()
    memory_service.observe_objects(
        [
            ObsObject(
                class_name="apple",
                track_id="apple-a",
                pose=(2.0, 0.0, 0.0),
                timestamp=10.0,
                confidence=0.9,
                room_id="kitchen",
            ),
            ObsObject(
                class_name="apple",
                track_id="apple-b",
                pose=(4.0, 0.0, 0.0),
                timestamp=11.0,
                confidence=0.9,
                room_id="kitchen",
            ),
        ]
    )
    service = MemoryPolicyService(
        memory_service,
        config=MemoryPolicyConfig(enabled=True, shadow_mode=False, ambiguity_gap_threshold=0.12),
    )

    context = service.build_context(
        instruction="아까 봤던 사과를 찾아가",
        target_class="apple",
        task_state="active",
        current_pose=(0.0, 0.0, 0.0),
        visible_target_now=False,
        room_id="kitchen",
    )
    decision, prompt_text = service.evaluate(context)

    assert context.candidate_count >= 2
    assert 0.0 <= context.retrieval_confidence <= 1.0
    assert decision.label in {MemoryPolicyLabel.ROUTE_MEMORY_VISION, MemoryPolicyLabel.WAIT}
    assert isinstance(decision.shadow_only, bool)
    assert prompt_text.endswith("Return exactly one label.")
