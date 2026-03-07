from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.sensors.d455_mount import D455AssetResolution
from runtime.bootstrap_profiles import profile_registry
from runtime.compatibility_report import build_compatibility_report


def test_compatibility_report_prefers_editor_assisted_for_runtime_mismatch(tmp_path: Path) -> None:
    isaac_root = tmp_path / "isaac-sim"
    apps_dir = isaac_root / "apps"
    apps_dir.mkdir(parents=True)
    (isaac_root / "warmup.bat").write_text("", encoding="utf-8")
    (isaac_root / "python.bat").write_text("", encoding="utf-8")
    (apps_dir / "isaacsim.exp.full.kit").write_text("", encoding="utf-8")
    (tmp_path / "python.exe").write_text("", encoding="utf-8")
    profile = profile_registry()["standalone_render_warmup"]

    report = build_compatibility_report(
        isaac_root=str(isaac_root),
        isaac_python=str(tmp_path / "python.exe"),
        selected_launch_mode="standalone_python",
        selected_profile=profile,
        asset_resolution=D455AssetResolution(
            assets_root="/Isaac",
            asset_path="/Isaac/Sensors/Intel/RealSense/rsd455.usd",
            exists=True,
            source="explicit",
        ),
        enabled_extensions=[],
        editor_available=True,
        extension_package_present=True,
    )

    assert report.likely_runtime_mismatch is True
    assert report.recommended_launch_mode == "editor_assisted"
    assert report.recommended_profile == "full_app_editor_assisted"


def test_compatibility_report_uses_explicit_experience_path(tmp_path: Path) -> None:
    isaac_root = tmp_path / "isaac-sim"
    isaac_root.mkdir(parents=True)
    (isaac_root / "python.bat").write_text("", encoding="utf-8")
    experience_path = tmp_path / "custom.kit"
    experience_path.write_text("", encoding="utf-8")
    profile = profile_registry()["full_app_editor_assisted"]

    report = build_compatibility_report(
        isaac_root=str(isaac_root),
        isaac_python=str(isaac_root / "python.bat"),
        selected_launch_mode="editor_assisted",
        selected_profile=profile,
        asset_resolution=D455AssetResolution(
            assets_root="/Isaac",
            asset_path="/Isaac/Sensors/Intel/RealSense/rsd455.usd",
            exists=True,
            source="explicit",
        ),
        enabled_extensions=["isaacsim.sensors.camera"],
        editor_available=True,
        extension_package_present=True,
        experience_path=str(experience_path),
    )

    assert report.experience_found is True
    assert report.launch_mode_supported is True
    assert report.context["experience_candidates"] == [str(experience_path)]
