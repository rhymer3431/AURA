from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from runtime.compatibility_report import CompatibilityReport
from runtime.smoke_result_model import SmokeResultSummary


@dataclass(frozen=True)
class Recommendation:
    action: str
    rationale: str
    severity: str = "info"
    recommended_launch_mode: str = ""
    recommended_profile: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_recommendations(
    *,
    compatibility_report: CompatibilityReport,
    failure_phase: str = "",
    smoke_result: SmokeResultSummary | None = None,
) -> list[Recommendation]:
    phase = str(failure_phase).strip().lower()
    results: list[Recommendation] = []
    for issue in compatibility_report.blocking_issues:
        results.append(
            Recommendation(
                action=issue,
                rationale="blocking_issue",
                severity="error",
                recommended_launch_mode=compatibility_report.recommended_launch_mode,
                recommended_profile=compatibility_report.recommended_profile,
            )
        )
    if not compatibility_report.isaac_python_found:
        results.append(
            Recommendation(
                action="Set ISAAC_SIM_ROOT or ISAAC_SIM_PYTHON to a valid Isaac installation and rerun preflight.",
                rationale="Isaac bundled python.bat is required for deprecated standalone live-smoke diagnostics.",
                severity="error",
                recommended_launch_mode="standalone_python",
                recommended_profile=compatibility_report.recommended_profile,
            )
        )
    if compatibility_report.likely_runtime_mismatch:
        results.append(
            Recommendation(
                action="Current interpreter is not Isaac python.bat. Use the deprecated live-smoke diagnostics path only if you are explicitly validating old bootstrap behavior.",
                rationale="likely_runtime_mismatch",
                severity="warning",
                recommended_launch_mode=compatibility_report.recommended_launch_mode,
                recommended_profile=compatibility_report.recommended_profile,
            )
        )
    if not compatibility_report.extension_mode_supported:
        results.append(
            Recommendation(
                action="Extension mode is unavailable until the repo extension package is enabled inside an Isaac Sim Full App / Kit process.",
                rationale="extension_mode_unavailable",
                severity="info",
                recommended_launch_mode="editor_assisted",
                recommended_profile="full_app_editor_assisted",
            )
        )
    if phase in {"simulation_app_created", "process_start"}:
        results.append(
            Recommendation(
                action="Verify Isaac install root, python.bat, and experience path. If standalone remains unstable, retry editor_assisted.",
                rationale="bootstrap failed before SimulationApp became usable",
                severity="warning",
                recommended_launch_mode="editor_assisted",
                recommended_profile="full_app_editor_assisted",
            )
        )
    if phase in {"assets_root_resolved", "d455_asset_resolved"}:
        results.append(
            Recommendation(
                action="Check Isaac assets root / Nucleus mount and confirm /Isaac/Sensors/Intel/RealSense/rsd455.usd is reachable.",
                rationale="D455 asset resolution failed",
                severity="warning",
            )
        )
    if phase in {
        "render_products_ready",
        "warmup_frames_completed",
        "first_rgb_frame_ready",
        "first_depth_frame_ready",
        "first_nonempty_frame_ready",
    }:
        results.append(
            Recommendation(
                action="Retry with bootstrap profile standalone_render_warmup or switch to editor_assisted for in-editor rendering warmup.",
                rationale="frame/annotator warmup did not complete",
                severity="warning",
                recommended_launch_mode="editor_assisted",
                recommended_profile="standalone_render_warmup",
            )
        )
    if smoke_result is not None:
        if smoke_result.pipeline_status.passed and not smoke_result.detections_nonempty:
            results.append(
                Recommendation(
                    action="Sensor and pipeline ingress succeeded, but detections were empty. Retry with a known target object in view or interpret this as an empty-scene pass.",
                    rationale="empty detection batch",
                    severity="info",
                    recommended_profile=compatibility_report.recommended_profile,
                )
            )
        if smoke_result.sensor_status.passed and not smoke_result.memory_status.passed:
            results.append(
                Recommendation(
                    action="Use --smoke-target-tier pipeline for ingress validation, or place a detectable object in view for memory-tier validation.",
                    rationale="memory tier did not pass after sensor/pipeline success",
                    severity="info",
                    recommended_profile=compatibility_report.recommended_profile,
                )
            )
    deduped: list[Recommendation] = []
    seen = set()
    for item in results:
        key = (item.action, item.rationale, item.recommended_launch_mode, item.recommended_profile)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
