from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adapters.sensors.d455_mount import D455AssetResolution
from runtime.bootstrap_profiles import (
    BootstrapProfile,
    PROFILE_EDITOR_ASSISTED,
    PROFILE_EXTENSION_IN_EDITOR,
    PROFILE_MINIMAL_HEADLESS_SENSOR,
    PROFILE_STANDALONE_RENDER_WARMUP,
)
from runtime.isaac_launch_modes import (
    LAUNCH_MODE_EDITOR_ASSISTED,
    LAUNCH_MODE_EXTENSION,
    LAUNCH_MODE_STANDALONE,
)


@dataclass(frozen=True)
class CompatibilityReport:
    isaac_root_found: bool
    isaac_python_found: bool
    experience_found: bool
    assets_root_found: bool
    d455_asset_found: bool
    required_extensions_available: bool
    launch_mode_supported: bool
    editor_assisted_supported: bool
    extension_mode_supported: bool
    warmup_scripts_available: bool
    likely_runtime_mismatch: bool
    recommended_profile: str
    recommended_launch_mode: str
    blocking_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        lines = [
            f"isaac_root_found={self.isaac_root_found}",
            f"isaac_python_found={self.isaac_python_found}",
            f"experience_found={self.experience_found}",
            f"assets_root_found={self.assets_root_found}",
            f"d455_asset_found={self.d455_asset_found}",
            f"launch_mode_supported={self.launch_mode_supported}",
            f"editor_assisted_supported={self.editor_assisted_supported}",
            f"extension_mode_supported={self.extension_mode_supported}",
            f"warmup_scripts_available={self.warmup_scripts_available}",
            f"recommended_profile={self.recommended_profile}",
            f"recommended_launch_mode={self.recommended_launch_mode}",
        ]
        lines.extend(f"warning={item}" for item in self.warnings)
        lines.extend(f"blocking_issue={item}" for item in self.blocking_issues)
        return lines


def build_compatibility_report(
    *,
    isaac_root: str,
    isaac_python: str,
    selected_launch_mode: str,
    selected_profile: BootstrapProfile,
    asset_resolution: D455AssetResolution,
    enabled_extensions: list[str],
    editor_available: bool,
    extension_package_present: bool,
    experience_path: str = "",
) -> CompatibilityReport:
    root_path = Path(str(isaac_root))
    python_path = Path(str(isaac_python))
    isaac_root_found = root_path.exists()
    isaac_python_found = python_path.exists()
    explicit_experience = Path(str(experience_path)).expanduser() if str(experience_path).strip() != "" else None
    if explicit_experience is not None and not explicit_experience.is_absolute():
        explicit_experience = root_path / explicit_experience
    experience_paths = [explicit_experience] if explicit_experience is not None else [root_path / rel_path for rel_path in selected_profile.required_experience]
    experience_found = any(path.exists() for path in experience_paths) if experience_paths else isaac_root_found
    assets_root_found = str(asset_resolution.assets_root).strip() != ""
    d455_asset_found = bool(asset_resolution.exists is True)
    warmup_scripts_available = (root_path / "warmup.bat").exists() or (root_path / "clear_caches.bat").exists()
    extension_names = {str(item) for item in enabled_extensions}
    required_extensions_available = not selected_profile.required_extensions or bool(extension_names) or bool(isaac_python_found)
    editor_assisted_supported = bool(editor_available)
    extension_mode_supported = bool(editor_available and extension_package_present)
    likely_runtime_mismatch = isaac_root_found and isaac_python_found and python_path.suffix.lower() not in {".bat", ".cmd"}

    blocking_issues: list[str] = []
    warnings: list[str] = []
    if not isaac_root_found:
        blocking_issues.append("Isaac install root was not found.")
    if not isaac_python_found:
        blocking_issues.append("Isaac bundled python.bat was not found.")
    if not experience_found:
        blocking_issues.append("Required Isaac experience file was not found for the selected bootstrap profile.")
    if not d455_asset_found:
        warnings.append("D455 asset could not be confirmed. Check assets root or Nucleus availability.")
    if not required_extensions_available:
        warnings.append("Required extensions could not be confirmed before app startup.")
    if likely_runtime_mismatch:
        warnings.append("Current Python executable is not Isaac python.bat; standalone runtime mismatch is likely.")
    if not warmup_scripts_available:
        warnings.append("Warmup/cache helper scripts were not found under the Isaac root.")
    if selected_launch_mode == LAUNCH_MODE_EDITOR_ASSISTED and not editor_assisted_supported:
        blocking_issues.append("editor_assisted requires execution inside a running Isaac Sim Full App / Kit process.")
    if selected_launch_mode == LAUNCH_MODE_EXTENSION and not extension_mode_supported:
        blocking_issues.append("extension_mode requires the repo extension package plus a running Isaac Sim Full App / Kit process.")

    recommended_launch_mode = str(selected_launch_mode)
    if selected_launch_mode == LAUNCH_MODE_EXTENSION and not extension_mode_supported:
        recommended_launch_mode = LAUNCH_MODE_EDITOR_ASSISTED if editor_assisted_supported else LAUNCH_MODE_STANDALONE
    elif selected_launch_mode == LAUNCH_MODE_EDITOR_ASSISTED and not editor_assisted_supported and isaac_python_found:
        recommended_launch_mode = LAUNCH_MODE_STANDALONE
    elif selected_launch_mode == LAUNCH_MODE_STANDALONE and likely_runtime_mismatch and editor_assisted_supported:
        recommended_launch_mode = LAUNCH_MODE_EDITOR_ASSISTED
    elif not isaac_python_found and editor_assisted_supported:
        recommended_launch_mode = LAUNCH_MODE_EDITOR_ASSISTED
    elif not experience_found and editor_assisted_supported:
        recommended_launch_mode = LAUNCH_MODE_EDITOR_ASSISTED

    if recommended_launch_mode == LAUNCH_MODE_EXTENSION:
        recommended_profile = PROFILE_EXTENSION_IN_EDITOR
    elif recommended_launch_mode == LAUNCH_MODE_EDITOR_ASSISTED:
        recommended_profile = PROFILE_EDITOR_ASSISTED
    elif selected_profile.smoke_target_tier == "sensor":
        recommended_profile = PROFILE_MINIMAL_HEADLESS_SENSOR
    else:
        recommended_profile = PROFILE_STANDALONE_RENDER_WARMUP

    return CompatibilityReport(
        isaac_root_found=isaac_root_found,
        isaac_python_found=isaac_python_found,
        experience_found=experience_found,
        assets_root_found=assets_root_found,
        d455_asset_found=d455_asset_found,
        required_extensions_available=required_extensions_available,
        launch_mode_supported=len(blocking_issues) == 0,
        editor_assisted_supported=editor_assisted_supported,
        extension_mode_supported=extension_mode_supported,
        warmup_scripts_available=warmup_scripts_available,
        likely_runtime_mismatch=likely_runtime_mismatch,
        recommended_profile=recommended_profile,
        recommended_launch_mode=recommended_launch_mode,
        blocking_issues=blocking_issues,
        warnings=warnings,
        context={
            "isaac_root": str(root_path),
            "isaac_python": str(python_path),
            "experience_candidates": [str(path) for path in experience_paths],
            "assets_root": str(asset_resolution.assets_root),
            "d455_asset_path": str(asset_resolution.asset_path),
            "enabled_extensions": list(enabled_extensions),
            "extension_package_present": bool(extension_package_present),
        },
    )
