from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime.aura_runtime import build_launch_config
from runtime.aura_runtime_args import (
    DEFAULT_NATIVE_VIEWER,
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
    assert args.viewer_publish is True
    assert args.native_viewer == "opencv"


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
    assert args.native_viewer == DEFAULT_NATIVE_VIEWER
    assert args.viewer_publish is False
    assert args.action_scale == 0.5
    assert args.obstacle_stop_distance_m == 0.45
    assert args.obstacle_hold_distance_m == 0.70
    assert args.obstacle_backoff_vx_mps == 0.18
    assert args.obstacle_recovery_hold_sec == 0.75


def test_build_arg_parser_accepts_skip_detection_flag():
    args = _parse_args("--skip-detection")

    assert args.skip_detection is True


def test_validate_args_rejects_native_viewer_without_viewer_publish():
    args = _parse_args("--native-viewer", "opencv")

    with pytest.raises(ValueError, match="--native-viewer opencv requires --viewer-publish"):
        validate_args(args)


def test_validate_args_rejects_obstacle_hold_distance_below_stop_distance():
    args = _parse_args("--obstacle-stop-distance-m", "0.5", "--obstacle-hold-distance-m", "0.4")

    with pytest.raises(
        ValueError,
        match="--obstacle-hold-distance-m must be greater than or equal to --obstacle-stop-distance-m",
    ):
        validate_args(args)


def test_build_launch_config_keeps_viewport_updates_when_viewer_publish_enabled():
    args = _parse_args("--headless", "--viewer-publish")

    apply_launch_mode_defaults(args)
    launch_config = build_launch_config(args)

    assert launch_config == {"headless": True}
