from __future__ import annotations

from dataclasses import dataclass


LAUNCH_MODE_STANDALONE = "standalone_python"
LAUNCH_MODE_ATTACH = "full_app_attach"
LAUNCH_MODE_EXTENSION = "extension_mode"
LAUNCH_MODE_AUTO = "auto"


@dataclass(frozen=True)
class LaunchModeAvailability:
    standalone_available: bool
    editor_available: bool


@dataclass(frozen=True)
class LaunchModeSelection:
    requested_mode: str
    selected_mode: str
    reason: str
    recommended_mode: str = ""


def normalize_launch_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "":
        return LAUNCH_MODE_AUTO
    return normalized


def select_launch_mode(requested_mode: str, *, availability: LaunchModeAvailability) -> LaunchModeSelection:
    requested = normalize_launch_mode(requested_mode)
    if requested == LAUNCH_MODE_STANDALONE:
        recommendation = LAUNCH_MODE_ATTACH if availability.editor_available else ""
        reason = "standalone Isaac Python bootstrap selected explicitly"
        if not availability.standalone_available:
            reason = "standalone Isaac Python bootstrap requested but isaacsim import is unavailable"
        return LaunchModeSelection(
            requested_mode=requested,
            selected_mode=LAUNCH_MODE_STANDALONE,
            reason=reason,
            recommended_mode=recommendation,
        )
    if requested in {LAUNCH_MODE_ATTACH, LAUNCH_MODE_EXTENSION}:
        return LaunchModeSelection(
            requested_mode=requested,
            selected_mode=requested,
            reason="attach/editor smoke selected explicitly",
            recommended_mode=LAUNCH_MODE_STANDALONE if availability.standalone_available else "",
        )
    if availability.editor_available:
        return LaunchModeSelection(
            requested_mode=requested,
            selected_mode=LAUNCH_MODE_ATTACH,
            reason="active Omniverse stage detected; attach smoke is preferred",
            recommended_mode=LAUNCH_MODE_STANDALONE if availability.standalone_available else "",
        )
    if availability.standalone_available:
        return LaunchModeSelection(
            requested_mode=requested,
            selected_mode=LAUNCH_MODE_STANDALONE,
            reason="isaacsim standalone bootstrap is available",
            recommended_mode=LAUNCH_MODE_ATTACH if availability.editor_available else "",
        )
    return LaunchModeSelection(
        requested_mode=requested,
        selected_mode=LAUNCH_MODE_STANDALONE,
        reason="no attach/editor context detected and isaacsim import is unavailable",
        recommended_mode="",
    )


def recommend_mode_for_failure(*, selected_mode: str, failure_phase: str) -> str:
    phase = str(failure_phase).strip().lower()
    mode = normalize_launch_mode(selected_mode)
    if mode == LAUNCH_MODE_STANDALONE and phase in {
        "simulation_app_created",
        "required_extensions_ready",
        "stage_ready",
        "d455_depth_sensor_initialized",
    }:
        return LAUNCH_MODE_ATTACH
    if mode in {LAUNCH_MODE_ATTACH, LAUNCH_MODE_EXTENSION} and phase in {"assets_root_resolved", "d455_asset_resolved"}:
        return LAUNCH_MODE_STANDALONE
    return ""
