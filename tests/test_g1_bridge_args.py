from __future__ import annotations

import pytest

from runtime.g1_bridge_args import (
    DEFAULT_OBJECT_SEARCH_INSTRUCTION,
    apply_demo_defaults,
    build_arg_parser,
    validate_args,
)


def _parse_args(*argv: str):
    return build_arg_parser().parse_args(list(argv))


def test_apply_demo_defaults_sets_object_search_instruction():
    args = _parse_args("--planner-mode", "dual", "--spawn-demo-object")

    apply_demo_defaults(args)

    assert args.instruction == DEFAULT_OBJECT_SEARCH_INSTRUCTION


def test_apply_demo_defaults_preserves_custom_instruction():
    args = _parse_args(
        "--planner-mode",
        "dual",
        "--spawn-demo-object",
        "--instruction",
        "Find the loading dock pallet and stop there.",
    )

    apply_demo_defaults(args)

    assert args.instruction == "Find the loading dock pallet and stop there."


def test_validate_args_rejects_demo_object_outside_dual_mode():
    args = _parse_args("--planner-mode", "pointgoal", "--goal-x", "2.0", "--goal-y", "0.0", "--spawn-demo-object")

    with pytest.raises(ValueError, match="--spawn-demo-object requires --planner-mode dual"):
        validate_args(args)


def test_validate_args_accepts_standard_pointgoal_mode():
    args = _parse_args("--planner-mode", "pointgoal", "--goal-x", "2.0", "--goal-y", "0.0")

    apply_demo_defaults(args)
    validate_args(args)


def test_validate_args_accepts_standard_dual_mode():
    args = _parse_args("--planner-mode", "dual", "--instruction", "Navigate to the target shelf and stop.")

    apply_demo_defaults(args)
    validate_args(args)


def test_validate_args_accepts_interactive_mode_without_goal_or_instruction():
    args = _parse_args("--planner-mode", "interactive")

    apply_demo_defaults(args)
    validate_args(args)


def test_validate_args_rejects_empty_interactive_prompt():
    args = _parse_args("--planner-mode", "interactive", "--interactive-prompt", "   ")

    with pytest.raises(ValueError, match="--interactive-prompt must be non-empty"):
        validate_args(args)
