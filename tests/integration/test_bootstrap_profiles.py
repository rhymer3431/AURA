from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime.bootstrap_profiles import (
    PROFILE_EDITOR_ASSISTED,
    PROFILE_MINIMAL_HEADLESS_SENSOR,
    PROFILE_STANDALONE_RENDER_WARMUP,
    select_bootstrap_profile,
)


def test_bootstrap_profile_selection_prefers_minimal_headless_sensor() -> None:
    selection = select_bootstrap_profile(
        "auto",
        launch_mode="standalone_python",
        headless=True,
        smoke_target_tier="sensor",
    )
    assert selection.selected_profile.name == PROFILE_MINIMAL_HEADLESS_SENSOR


def test_bootstrap_profile_selection_prefers_editor_profile_in_editor_mode() -> None:
    selection = select_bootstrap_profile(
        "auto",
        launch_mode="editor_assisted",
        headless=False,
        smoke_target_tier="memory",
    )
    assert selection.selected_profile.name == PROFILE_EDITOR_ASSISTED


def test_bootstrap_profile_selection_uses_render_warmup_for_memory_target() -> None:
    selection = select_bootstrap_profile(
        "auto",
        launch_mode="standalone_python",
        headless=True,
        smoke_target_tier="memory",
    )
    assert selection.selected_profile.name == PROFILE_STANDALONE_RENDER_WARMUP
