from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from systems.transport.messages import TaskRequest
from server.execution_mode_classifier import ExecutionModeClassifier


def test_classifier_routes_memory_cue_to_mem_nav() -> None:
    classifier = ExecutionModeClassifier()

    result = classifier.classify(TaskRequest(command_text="아까 봤던 사과를 찾아가"))

    assert result.mode == "MEM_NAV"
    assert result.reason == "memory_cue"
    assert result.target_class == "apple"


def test_classifier_routes_general_embodied_instruction_to_nav() -> None:
    classifier = ExecutionModeClassifier()

    result = classifier.classify(TaskRequest(command_text="loading dock로 가"))

    assert result.mode == "NAV"
    assert result.reason == "embodied_instruction"
    assert result.target_class == ""


def test_classifier_keeps_pure_question_in_talk_mode() -> None:
    classifier = ExecutionModeClassifier()

    result = classifier.classify(TaskRequest(command_text="왜 멈췄어"))

    assert result.mode == "TALK"
    assert result.reason == "conversation"
