from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime.compatibility_report import CompatibilityReport
from runtime.recommendation_engine import build_recommendations
from runtime.smoke_result_model import aggregate_smoke_result


def _compatibility() -> CompatibilityReport:
    return CompatibilityReport(
        isaac_root_found=True,
        isaac_python_found=True,
        experience_found=True,
        assets_root_found=True,
        d455_asset_found=True,
        required_extensions_available=True,
        launch_mode_supported=True,
        editor_assisted_supported=True,
        extension_mode_supported=True,
        warmup_scripts_available=True,
        likely_runtime_mismatch=False,
        recommended_profile="standalone_render_warmup",
        recommended_launch_mode="editor_assisted",
    )


def test_recommendation_engine_suggests_warmup_for_frame_phase_failure() -> None:
    recommendations = build_recommendations(
        compatibility_report=_compatibility(),
        failure_phase="first_rgb_frame_ready",
    )
    assert any("standalone_render_warmup" in item.action for item in recommendations)


def test_recommendation_engine_explains_empty_detection_batch() -> None:
    smoke_result = aggregate_smoke_result(
        target_tier="memory",
        frame_received=True,
        rgb_ready=True,
        depth_ready=True,
        pose_ready=True,
        detection_attempted=True,
        detections_nonempty=False,
        perception_ingress_ready=True,
        memory_ingress_ready=True,
        memory_update_called=False,
    )
    recommendations = build_recommendations(
        compatibility_report=_compatibility(),
        failure_phase="",
        smoke_result=smoke_result,
    )
    assert any("empty-scene pass" in item.action for item in recommendations)
