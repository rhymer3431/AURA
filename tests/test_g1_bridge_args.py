from __future__ import annotations

import pytest

from runtime.g1_bridge import build_launch_config
from runtime.g1_bridge_args import (
    DEFAULT_OBJECT_SEARCH_INSTRUCTION,
    DEFAULT_VIEWER_CONTROL_ENDPOINT,
    DEFAULT_VIEWER_SHM_NAME,
    apply_demo_defaults,
    apply_launch_mode_defaults,
    build_arg_parser,
    resolve_launch_mode,
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


def test_apply_launch_mode_defaults_forces_gui_over_headless_flag():
    args = _parse_args("--launch-mode", "gui", "--headless")

    apply_launch_mode_defaults(args)

    assert args.resolved_launch_mode == "gui"
    assert args.headless is False


def test_apply_launch_mode_defaults_forces_headless_for_g1_view():
    args = _parse_args("--launch-mode", "g1_view")

    apply_launch_mode_defaults(args)

    assert args.resolved_launch_mode == "g1_view"
    assert args.headless is True


def test_build_launch_config_keeps_viewport_updates_for_g1_view():
    args = _parse_args("--launch-mode", "g1_view")

    apply_launch_mode_defaults(args)
    launch_config = build_launch_config(args)

    assert launch_config == {"headless": True}


def test_resolve_launch_mode_keeps_legacy_headless_behavior_when_unspecified():
    args = _parse_args("--headless")

    assert resolve_launch_mode(args) == "headless"


def test_build_arg_parser_exposes_viewer_transport_defaults():
    args = _parse_args()

    assert args.viewer_control_endpoint == DEFAULT_VIEWER_CONTROL_ENDPOINT
    assert args.viewer_shm_name == DEFAULT_VIEWER_SHM_NAME


def test_build_arg_parser_accepts_skip_detection_flag():
    args = _parse_args("--skip-detection")

    assert args.skip_detection is True
