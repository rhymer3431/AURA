from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from adapters.sensors.d455_mount import (
    DEFAULT_D455_MOUNT_PRIM_PATH,
    dump_prim_tree,
    ensure_d455_mount,
    resolve_d455_asset_path,
)
from adapters.sensors.d455_sensor import D455SensorAdapter, D455SensorAdapterConfig
from adapters.sensors.isaac_live_source import IsaacLiveFrameSource, IsaacLiveSourceConfig
from apps.runtime_common import frame_sample_to_batch
from locomotion.paths import resolve_environment_reference
from locomotion.scene import spawn_environment
from runtime.bootstrap_diagnostics import BootstrapPhaseTracker
from runtime.bootstrap_profiles import BootstrapProfileSelection, select_bootstrap_profile
from runtime.compatibility_report import CompatibilityReport, build_compatibility_report
from runtime.isaac_launch_modes import (
    LAUNCH_MODE_AUTO,
    LAUNCH_MODE_EDITOR_ASSISTED,
    LAUNCH_MODE_EXTENSION,
    LAUNCH_MODE_STANDALONE,
    LaunchModeAvailability,
    LaunchModeSelection,
    recommend_mode_for_failure,
    select_launch_mode,
)
from runtime.recommendation_engine import build_recommendations
from runtime.smoke_result_model import SmokeResultSummary, aggregate_smoke_result
from runtime.supervisor import Supervisor, SupervisorConfig
from services.memory_service import MemoryService


@dataclass(frozen=True)
class LiveSmokePhaseTimeouts:
    app_bootstrap_sec: float = 90.0
    stage_ready_sec: float = 45.0
    sensor_init_sec: float = 45.0
    first_frame_sec: float = 20.0

    def as_dict(self) -> dict[str, float]:
        return {
            "simulation_app_created": float(self.app_bootstrap_sec),
            "required_extensions_ready": float(self.app_bootstrap_sec),
            "stage_ready": float(self.stage_ready_sec),
            "stage_opened_or_created": float(self.stage_ready_sec),
            "assets_root_resolved": float(self.stage_ready_sec),
            "d455_asset_resolved": float(self.stage_ready_sec),
            "d455_prim_spawned": float(self.sensor_init_sec),
            "d455_reference_bound": float(self.sensor_init_sec),
            "sensor_wrapper_created": float(self.sensor_init_sec),
            "d455_depth_sensor_initialized": float(self.sensor_init_sec),
            "warmup_frames_started": float(self.sensor_init_sec),
            "warmup_frames_completed": float(self.sensor_init_sec),
            "annotators_ready": float(self.sensor_init_sec),
            "render_products_ready": float(self.sensor_init_sec),
            "first_rgb_frame_ready": float(self.first_frame_sec),
            "first_depth_frame_ready": float(self.first_frame_sec),
            "first_nonempty_frame_ready": float(self.first_frame_sec),
            "first_pose_ready": float(self.first_frame_sec),
            "observation_batch_processed": float(self.first_frame_sec),
            "perception_ingress_ready": float(self.first_frame_sec),
            "memory_updated": float(self.first_frame_sec),
            "memory_ingress_ready": float(self.first_frame_sec),
            "sensor_smoke_pass": float(self.first_frame_sec),
            "pipeline_smoke_pass": float(self.first_frame_sec),
            "memory_smoke_pass": float(self.first_frame_sec),
            "full_smoke_pass": float(self.first_frame_sec),
            "smoke_pass": float(self.first_frame_sec),
        }


class SmokeMemoryService(MemoryService):
    def __init__(self) -> None:
        super().__init__(db_path=None)
        self.observe_call_count = 0
        self.observe_item_count = 0
        self.update_call_count = 0

    def observe_objects(self, observations) -> list[object]:  # noqa: ANN001
        observation_list = list(observations)
        self.observe_call_count += 1
        self.observe_item_count += len(observation_list)
        return super().observe_objects(observation_list)

    def update_from_observation(self, observation) -> object:  # noqa: ANN001
        self.update_call_count += 1
        return super().update_from_observation(observation)


class LiveSmokeRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        tracker: BootstrapPhaseTracker | None = None,
        simulation_app_factory: Callable[[dict[str, object]], object] | None = None,
        world_factory: Callable[..., object] | None = None,
        sensor_factory: Callable[[D455SensorAdapterConfig], D455SensorAdapter] | None = None,
        supervisor_factory: Callable[..., Supervisor] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.args = args
        self.time_fn = time_fn or time.monotonic
        self.phase_timeouts = LiveSmokePhaseTimeouts(
            app_bootstrap_sec=float(getattr(args, "app_bootstrap_timeout_sec", 90.0)),
            stage_ready_sec=float(getattr(args, "stage_ready_timeout_sec", 45.0)),
            sensor_init_sec=float(getattr(args, "sensor_init_timeout_sec", 45.0)),
            first_frame_sec=float(getattr(args, "first_frame_timeout_sec", 20.0)),
        )
        artifact_dir = Path(str(getattr(args, "artifacts_dir", "tmp/process_logs/live_smoke"))).expanduser()
        diagnostics_path = Path(str(getattr(args, "diagnostics_path", artifact_dir / "diagnostics.json"))).expanduser()
        self.tracker = tracker or BootstrapPhaseTracker(
            diagnostics_path=diagnostics_path,
            artifact_dir=artifact_dir,
            launch_mode=str(getattr(args, "launch_mode", LAUNCH_MODE_AUTO)),
            frame_source="live",
            headless=bool(getattr(args, "headless", False)),
            cli_args=list(getattr(args, "_argv", [])),
            phase_timeouts=self.phase_timeouts.as_dict(),
        )
        self._simulation_app_factory = simulation_app_factory
        self._world_factory = world_factory
        self._sensor_factory = sensor_factory or (lambda cfg: D455SensorAdapter(cfg))
        self._supervisor_factory = supervisor_factory or self._default_supervisor_factory
        self._selected_launch: LaunchModeSelection | None = None
        self._selected_profile: BootstrapProfileSelection | None = None
        self._compatibility_report: CompatibilityReport | None = None
        self._extension_package_present = _extension_package_present()

    def run_preflight(self) -> int:
        self._mark_process_start(mode="preflight")
        try:
            self._record_env_resolution()
            launch_selection = self._select_launch_mode()
            profile_selection = self._select_profile(launch_selection.selected_mode)
            self._record_extension_state(simulation_app=None)
            asset_resolution = self._resolve_d455_asset()
            compatibility = self._build_and_store_compatibility(asset_resolution=asset_resolution)
            self._emit_recommendations(compatibility_report=compatibility, failure_phase="")
            if compatibility.blocking_issues:
                raise RuntimeError("; ".join(compatibility.blocking_issues))
            self.tracker.finalize_success("preflight completed")
            self.tracker.write_json_artifact(
                "preflight_summary",
                filename="preflight_summary.json",
                payload={
                    "selected_launch_mode": launch_selection.selected_mode,
                    "selected_profile": profile_selection.selected_profile.name,
                    "selected_profile_reason": profile_selection.reason,
                    "target_tier": self._smoke_target_tier(),
                    "d455_asset_path": asset_resolution.asset_path,
                    "d455_asset_exists": asset_resolution.exists,
                    "recommended_launch_mode": compatibility.recommended_launch_mode,
                    "recommended_profile": compatibility.recommended_profile,
                },
            )
            self._print_compatibility(compatibility)
            self._print_summary()
            return 0
        except Exception as exc:  # noqa: BLE001
            failure_phase = self.tracker.diagnostics.failure_phase or self.tracker.diagnostics.current_phase or "isaac_python_env_resolved"
            self.tracker.fail_phase(failure_phase, exc=exc)
            return self._finalize_failure_with_recommendations(failure_phase=failure_phase)

    def run_smoke(self) -> int:
        launch_selection = self._select_launch_mode()
        self._select_profile(launch_selection.selected_mode)
        if launch_selection.selected_mode == LAUNCH_MODE_STANDALONE:
            return self._run_standalone_smoke(selection_reason=launch_selection.reason)
        return self.run_in_editor(launch_mode=launch_selection.selected_mode)

    def run_in_editor(
        self,
        *,
        simulation_app=None,
        stage=None,
        launch_mode: str = LAUNCH_MODE_EDITOR_ASSISTED,
    ) -> int:
        self._mark_process_start(mode="editor_smoke")
        self.tracker.set_runtime_context(launch_mode=str(launch_mode))
        try:
            self._record_env_resolution()
            self._selected_launch = LaunchModeSelection(
                requested_mode=str(launch_mode),
                selected_mode=str(launch_mode),
                reason="in-editor smoke selected explicitly",
            )
            self._select_profile(str(launch_mode))
            resolved_stage, resolved_app = self._resolve_editor_stage(
                stage=stage,
                simulation_app=simulation_app,
                launch_mode=str(launch_mode),
            )
            asset_resolution = self._resolve_d455_asset()
            compatibility = self._build_and_store_compatibility(asset_resolution=asset_resolution)
            if compatibility.blocking_issues:
                raise RuntimeError("; ".join(compatibility.blocking_issues))
            return self._run_smoke_pipeline(simulation_app=resolved_app, stage=resolved_stage, world=None)
        except Exception as exc:  # noqa: BLE001
            failure_phase = self.tracker.diagnostics.failure_phase or self.tracker.diagnostics.current_phase or "stage_ready"
            self.tracker.fail_phase(failure_phase, exc=exc)
            return self._finalize_failure_with_recommendations(failure_phase=failure_phase)

    def _run_standalone_smoke(self, *, selection_reason: str) -> int:
        self._mark_process_start(mode="smoke")
        simulation_app = None
        try:
            self._record_env_resolution()
            asset_resolution = self._resolve_d455_asset()
            compatibility = self._build_and_store_compatibility(asset_resolution=asset_resolution)
            if compatibility.blocking_issues:
                raise RuntimeError("; ".join(compatibility.blocking_issues))
            simulation_app = self._create_simulation_app()
            stage, world = self._prepare_stage(simulation_app)
            self.tracker.set_runtime_context(selected_launch_reason=selection_reason)
            compatibility = self._build_and_store_compatibility(asset_resolution=asset_resolution)
            if compatibility.blocking_issues:
                raise RuntimeError("; ".join(compatibility.blocking_issues))
            return self._run_smoke_pipeline(simulation_app=simulation_app, stage=stage, world=world)
        except Exception as exc:  # noqa: BLE001
            failure_phase = self.tracker.diagnostics.failure_phase or self.tracker.diagnostics.current_phase or "simulation_app_created"
            self.tracker.fail_phase(failure_phase, exc=exc)
            return self._finalize_failure_with_recommendations(failure_phase=failure_phase)
        finally:
            if simulation_app is not None:
                close = getattr(simulation_app, "close", None)
                if callable(close):
                    close()

    def _run_smoke_pipeline(self, *, simulation_app, stage, world) -> int:  # noqa: ANN001
        asset_path = ""
        if self._compatibility_report is not None:
            asset_path = str(self._compatibility_report.context.get("d455_asset_path", ""))
        mount_report = self._mount_d455(stage, asset_path)
        sensor = self._initialize_sensor(simulation_app, stage)
        self._warmup_runtime(simulation_app=simulation_app, world=world)
        render_context = sensor.diagnostics_snapshot()
        self.tracker.start_phase("annotators_ready", context=render_context)
        if render_context.get("camera_prim_paths"):
            self.tracker.succeed_phase("annotators_ready", context=render_context)
        else:
            self.tracker.skip_phase("annotators_ready", reason="Camera prims were discovered but annotator state could not be confirmed.", context=render_context)
        self.tracker.start_phase("render_products_ready", context=render_context)
        self.tracker.write_json_artifact("sensor_diagnostics", filename="sensor_diagnostics.json", payload=render_context)
        if render_context.get("render_product_paths"):
            self.tracker.succeed_phase("render_products_ready", context=render_context)
        else:
            self.tracker.fail_phase(
                "render_products_ready",
                message="Camera initialized but render product paths were not discovered.",
                context=render_context,
            )
            return self._finalize_failure_with_recommendations(failure_phase="render_products_ready")

        sample = self._capture_first_sample(simulation_app, stage, sensor)
        smoke_result = self._process_sample(sample)
        self._mark_tier_phases(smoke_result)
        self._emit_recommendations(
            compatibility_report=self._compatibility_report,
            failure_phase="",
            smoke_result=smoke_result,
        )
        mount_context = {"mount_report": mount_report.as_dict()}
        self.tracker.start_phase("smoke_pass", context={**mount_context, **smoke_result.as_dict()})
        success_states = {"sensor_smoke_pass", "pipeline_smoke_pass", "memory_smoke_pass", "full_smoke_pass"}
        if smoke_result.overall_status in success_states:
            self.tracker.succeed_phase("smoke_pass", context={**mount_context, **smoke_result.as_dict()})
            self.tracker.finalize_success(smoke_result.overall_status)
            if self._compatibility_report is not None:
                self._print_compatibility(self._compatibility_report)
            self._print_summary()
            return 0
        self.tracker.fail_phase("smoke_pass", message=smoke_result.overall_status, context={**mount_context, **smoke_result.as_dict()})
        return self._finalize_failure_with_recommendations(failure_phase="smoke_pass", smoke_result=smoke_result)

    def _mark_process_start(self, *, mode: str) -> None:
        self.tracker.start_phase("process_start", context={"mode": mode})
        self.tracker.succeed_phase("process_start", context={"pid": _safe_pid()})

    def _select_launch_mode(self) -> LaunchModeSelection:
        availability = LaunchModeAvailability(
            standalone_available=self._standalone_available(),
            editor_available=self._editor_attach_available(),
        )
        selection = select_launch_mode(str(getattr(self.args, "launch_mode", LAUNCH_MODE_AUTO)), availability=availability)
        self._selected_launch = selection
        launch_mode_alias = ""
        raw_requested = str(getattr(self.args, "launch_mode", "")).strip().lower()
        if selection.deprecated_alias_used:
            launch_mode_alias = raw_requested
        self.tracker.set_runtime_context(
            launch_mode=selection.selected_mode,
            launch_mode_alias=launch_mode_alias,
            selected_launch_reason=selection.reason,
            context={
                "requested_launch_mode": selection.requested_mode,
                "recommended_mode": selection.recommended_mode,
                "standalone_available": availability.standalone_available,
                "editor_available": availability.editor_available,
                "deprecated_launch_mode_alias_used": bool(selection.deprecated_alias_used),
            },
        )
        return selection

    def _ensure_profile_selection(self, launch_mode: str | None = None) -> BootstrapProfileSelection:
        if self._selected_profile is None:
            self._selected_profile = select_bootstrap_profile(
                str(getattr(self.args, "bootstrap_profile", "auto")),
                launch_mode=str(launch_mode or getattr(self.args, "launch_mode", LAUNCH_MODE_STANDALONE)),
                headless=bool(getattr(self.args, "headless", False)),
                smoke_target_tier=self._requested_smoke_target_tier(),
            )
        return self._selected_profile

    def _select_profile(self, launch_mode: str) -> BootstrapProfileSelection:
        selection = self._ensure_profile_selection(launch_mode)
        self.tracker.set_runtime_context(
            selected_profile=selection.selected_profile.name,
            selected_profile_reason=selection.reason,
            smoke_target_tier=self._smoke_target_tier(),
        )
        timeout_sec = float(self._profile_value("first_frame_timeout_sec"))
        for phase_name in ("first_rgb_frame_ready", "first_depth_frame_ready", "first_nonempty_frame_ready", "first_pose_ready"):
            self.tracker.update_phase_timeout(phase_name, timeout_sec)
        return selection

    def _record_env_resolution(self) -> None:
        self.tracker.start_phase("isaac_python_env_resolved")
        isaac_root = _discover_isaac_root()
        isaac_python = _discover_isaac_python()
        root_path = Path(isaac_root)
        self.tracker.set_runtime_context(
            script_path=str(Path(sys.argv[0]).resolve()),
            isaac_root=isaac_root,
            isaac_python=isaac_python,
            python_executable=str(sys.executable),
        )
        self.tracker.write_json_artifact(
            "cli_args",
            filename="cli_args.json",
            payload={"argv": list(getattr(self.args, "_argv", []))},
        )
        self.tracker.succeed_phase(
            "isaac_python_env_resolved",
            context={
                "script_path": str(Path(sys.argv[0]).resolve()),
                "isaac_root": isaac_root,
                "isaac_python": isaac_python,
                "python_executable": str(sys.executable),
                "experience_path": str(getattr(self.args, "experience_path", "")),
                "clear_cache_script": str(root_path / "clear_caches.bat"),
                "clear_cache_script_exists": (root_path / "clear_caches.bat").exists(),
                "warmup_script": str(root_path / "warmup.bat"),
                "warmup_script_exists": (root_path / "warmup.bat").exists(),
                "headless": bool(getattr(self.args, "headless", False)),
                "selected_frame_source": "live",
                "selected_launch_mode": str(self.tracker.diagnostics.launch_mode),
            },
        )

    def _record_extension_state(self, simulation_app) -> None:  # noqa: ANN001
        self.tracker.start_phase("required_extensions_ready")
        if simulation_app is not None:
            _call_update(simulation_app)
        extensions = _enabled_extensions()
        self.tracker.set_runtime_context(enabled_extensions=extensions)
        self.tracker.write_json_artifact(
            "enabled_extensions",
            filename="enabled_extensions.json",
            payload={"enabled_extensions": extensions},
        )
        self.tracker.succeed_phase("required_extensions_ready", context={"enabled_extensions": extensions})

    def _build_and_store_compatibility(self, *, asset_resolution) -> CompatibilityReport:
        selected_launch_mode = (
            self._selected_launch.selected_mode
            if self._selected_launch is not None
            else str(self.tracker.diagnostics.launch_mode or getattr(self.args, "launch_mode", LAUNCH_MODE_STANDALONE))
        )
        selected_profile = self._ensure_profile_selection(selected_launch_mode).selected_profile
        compatibility = build_compatibility_report(
            isaac_root=str(self.tracker.diagnostics.isaac_root or _discover_isaac_root()),
            isaac_python=str(self.tracker.diagnostics.isaac_python or _discover_isaac_python()),
            selected_launch_mode=selected_launch_mode,
            selected_profile=selected_profile,
            asset_resolution=asset_resolution,
            enabled_extensions=list(self.tracker.diagnostics.enabled_extensions),
            editor_available=self._editor_attach_available(),
            extension_package_present=self._extension_package_present,
            experience_path=str(getattr(self.args, "experience_path", "")),
        )
        self._compatibility_report = compatibility
        self.tracker.set_compatibility_report(compatibility.as_dict())
        self.tracker.write_json_artifact(
            "compatibility_report",
            filename="compatibility_report.json",
            payload=compatibility.as_dict(),
        )
        return compatibility

    def _create_simulation_app(self):
        self.tracker.start_phase("simulation_app_created")
        launch_config = {"headless": bool(getattr(self.args, "headless", False))}
        if bool(getattr(self.args, "headless", False)):
            launch_config["disable_viewport_updates"] = True
        if self._simulation_app_factory is not None:
            simulation_app = self._simulation_app_factory(dict(launch_config))
        else:
            from isaacsim import SimulationApp

            simulation_app = SimulationApp(launch_config=launch_config)
        self.tracker.succeed_phase("simulation_app_created", context={"launch_config": launch_config})
        self._record_extension_state(simulation_app)
        return simulation_app

    def _prepare_stage(self, simulation_app):
        self.tracker.start_phase("stage_ready")
        if self._world_factory is not None:
            world = self._world_factory()
        else:
            from isaacsim.core.api import World

            rendering_dt = max(float(getattr(self.args, "rendering_dt", 0.0)), float(getattr(self.args, "physics_dt", 1.0 / 60.0)))
            world = World(
                stage_units_in_meters=1.0,
                physics_dt=float(getattr(self.args, "physics_dt", 1.0 / 60.0)),
                rendering_dt=rendering_dt,
            )
        self.tracker.start_phase("stage_opened_or_created")
        env_reference = resolve_environment_reference(getattr(self.args, "scene_usd", None), str(getattr(self.args, "env_url", "")))
        spawn_environment(
            env_reference,
            str(getattr(self.args, "scene_prim_path", "/World/Environment")),
            tuple(getattr(self.args, "scene_translate", (0.0, 0.0, 0.0))),
        )
        for _ in range(max(int(self._profile_value("stage_settle_updates")), 0)):
            _call_update(simulation_app)
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable after SimulationApp bootstrap.")
        prim_tree = dump_prim_tree(stage)
        self.tracker.write_artifact("stage_prim_tree", filename="stage_prim_tree.txt", payload=prim_tree)
        stage_context = {
            "env_reference": env_reference,
            "scene_prim_path": str(getattr(self.args, "scene_prim_path", "/World/Environment")),
        }
        self.tracker.succeed_phase("stage_opened_or_created", context=stage_context)
        self.tracker.succeed_phase("stage_ready", context=stage_context)
        return stage, world

    def _resolve_editor_stage(self, *, stage=None, simulation_app=None, launch_mode: str):
        self.tracker.start_phase("stage_ready")
        try:
            import omni.usd
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("editor_assisted smoke requires execution inside Isaac Sim/Kit with omni.usd available.") from exc
        resolved_stage = stage if stage is not None else omni.usd.get_context().get_stage()
        if resolved_stage is None:
            raise RuntimeError("No active USD stage found. Start Isaac Sim Full App first, then retry in-editor smoke.")
        if simulation_app is None:
            try:
                import omni.kit.app
            except Exception:  # noqa: BLE001
                resolved_app = SimpleNamespace(update=lambda: None)
                kit_context = "omni.kit.app unavailable"
            else:
                resolved_app = omni.kit.app.get_app() or SimpleNamespace(update=lambda: None)
                kit_context = str(type(resolved_app).__name__)
        else:
            resolved_app = simulation_app
            kit_context = str(type(resolved_app).__name__)
        self.tracker.start_phase("stage_opened_or_created")
        prim_tree = dump_prim_tree(resolved_stage)
        self.tracker.write_artifact("stage_prim_tree", filename="stage_prim_tree.txt", payload=prim_tree)
        root_layer_path = ""
        root_layer = getattr(resolved_stage, "GetRootLayer", None)
        if callable(root_layer):
            try:
                layer = root_layer()
            except Exception:  # noqa: BLE001
                layer = None
            if layer is not None:
                root_layer_path = str(getattr(layer, "identifier", getattr(layer, "realPath", "")) or "")
        context = {
            "app_already_running": True,
            "current_stage_path": root_layer_path,
            "current_kit_context": kit_context,
            "launch_mode": str(launch_mode),
        }
        self.tracker.succeed_phase("stage_opened_or_created", context=context)
        self.tracker.succeed_phase("stage_ready", context=context)
        self._record_extension_state(resolved_app)
        return resolved_stage, resolved_app

    def _resolve_d455_asset(self):
        self.tracker.start_phase("assets_root_resolved")
        asset_resolution = resolve_d455_asset_path(
            asset_path=str(getattr(self.args, "d455_asset_path", "")).strip(),
            assets_root=str(getattr(self.args, "assets_root", "")).strip(),
        )
        self.tracker.succeed_phase(
            "assets_root_resolved",
            context={
                "assets_root": asset_resolution.assets_root,
                "source": asset_resolution.source,
                "message": asset_resolution.message,
            },
        )
        self.tracker.start_phase("d455_asset_resolved")
        if asset_resolution.exists is False:
            raise FileNotFoundError(f"D455 asset not found: {asset_resolution.asset_path}")
        self.tracker.succeed_phase(
            "d455_asset_resolved",
            context={
                "d455_asset_path": asset_resolution.asset_path,
                "d455_asset_exists": asset_resolution.exists,
            },
        )
        self.tracker.write_json_artifact(
            "d455_asset_resolution",
            filename="d455_asset_resolution.json",
            payload={
                "assets_root": asset_resolution.assets_root,
                "asset_path": asset_resolution.asset_path,
                "exists": asset_resolution.exists,
                "source": asset_resolution.source,
                "message": asset_resolution.message,
            },
        )
        return asset_resolution

    def _mount_d455(self, stage, asset_path: str):
        self.tracker.start_phase("d455_prim_spawned")
        mount_report = ensure_d455_mount(
            stage,
            asset_path=asset_path,
            prim_path=str(getattr(self.args, "d455_prim_path", DEFAULT_D455_MOUNT_PRIM_PATH)),
        )
        self.tracker.write_json_artifact("d455_mount_report", filename="d455_mount_report.json", payload=mount_report.as_dict())
        if not mount_report.prim_exists:
            raise RuntimeError(f"D455 prim mount failed at {mount_report.prim_path}")
        self.tracker.succeed_phase("d455_prim_spawned", context=mount_report.as_dict())
        self.tracker.start_phase("d455_reference_bound", context=mount_report.as_dict())
        self.tracker.succeed_phase("d455_reference_bound", context=mount_report.as_dict())
        return mount_report

    def _initialize_sensor(self, simulation_app, stage):
        self.tracker.start_phase("sensor_wrapper_created")
        sensor = self._sensor_factory(
            D455SensorAdapterConfig(
                use_d455=True,
                image_width=int(getattr(self.args, "image_width", 640)),
                image_height=int(getattr(self.args, "image_height", 640)),
                depth_max_m=float(getattr(self.args, "depth_max_m", 5.0)),
                strict_d455=True,
                force_runtime_mount=bool(getattr(self.args, "force_runtime_camera", False)),
            )
        )
        self.tracker.succeed_phase("sensor_wrapper_created", context={"sensor_factory": str(type(sensor).__name__)})
        self.tracker.start_phase("d455_depth_sensor_initialized")
        attempts = max(int(self._profile_value("sensor_init_retries")), 0) + 1
        last_message = ""
        for attempt in range(1, attempts + 1):
            ok, message = sensor.initialize(simulation_app, stage)
            last_message = str(message)
            if ok:
                sensor_report = sensor.diagnostics_snapshot()
                sensor_report["initialize_ok"] = bool(ok)
                sensor_report["initialize_message"] = str(message)
                sensor_report["attempt"] = attempt
                self.tracker.write_json_artifact("sensor_init_report", filename="sensor_init_report.json", payload=sensor_report)
                self.tracker.succeed_phase("d455_depth_sensor_initialized", context=sensor_report)
                return sensor
            for _ in range(max(int(self._profile_value("sensor_init_retry_updates")), 0)):
                _call_update(simulation_app)
        sensor_report = sensor.diagnostics_snapshot()
        sensor_report["initialize_ok"] = False
        sensor_report["initialize_message"] = last_message
        sensor_report["attempts"] = attempts
        self.tracker.write_json_artifact("sensor_init_report", filename="sensor_init_report.json", payload=sensor_report)
        raise RuntimeError(last_message or "D455 sensor initialize failed.")

    def _warmup_runtime(self, *, simulation_app, world) -> None:  # noqa: ANN001
        render_updates = max(int(self._profile_value("render_warmup_updates")), 0)
        physics_steps = max(int(self._profile_value("physics_warmup_steps")), 0)
        stage_updates = max(int(self._profile_value("stage_settle_updates")), 0)
        self.tracker.start_phase("warmup_frames_started")
        self.tracker.succeed_phase(
            "warmup_frames_started",
            context={
                "render_warmup_updates": render_updates,
                "physics_warmup_steps": physics_steps,
                "stage_settle_updates": stage_updates,
            },
        )
        for _ in range(stage_updates):
            _call_update(simulation_app)
        for _ in range(render_updates):
            _call_update(simulation_app)
        if world is not None:
            step = getattr(world, "step", None)
            if callable(step):
                for _ in range(physics_steps):
                    step(render=not bool(getattr(self.args, "headless", False)))
        self.tracker.start_phase("warmup_frames_completed")
        self.tracker.succeed_phase(
            "warmup_frames_completed",
            context={
                "render_warmup_updates": render_updates,
                "physics_warmup_steps": physics_steps,
                "stage_settle_updates": stage_updates,
            },
        )

    def _capture_first_sample(self, simulation_app, stage, sensor: D455SensorAdapter):
        frame_source = IsaacLiveFrameSource(
            simulation_app=simulation_app,
            stage=stage,
            sensor_adapter=sensor,
            robot_pose_provider=lambda: (0.0, 0.0, 0.0),
            robot_yaw_provider=lambda: 0.0,
            config=IsaacLiveSourceConfig(
                source_name="live_smoke",
                strict_live=True,
                image_width=int(getattr(self.args, "image_width", 640)),
                image_height=int(getattr(self.args, "image_height", 640)),
                depth_max_m=float(getattr(self.args, "depth_max_m", 5.0)),
            ),
        )
        source_report = frame_source.start()
        self.tracker.write_json_artifact(
            "frame_source_report",
            filename="frame_source_report.json",
            payload={
                "status": source_report.status,
                "source_name": source_report.source_name,
                "fallback_used": source_report.fallback_used,
                "notice": source_report.notice,
                "details": dict(source_report.details),
            },
        )
        if source_report.status != "ready":
            raise RuntimeError(source_report.notice or "Live frame source did not become ready.")
        sample = self._wait_for_first_sample(frame_source)
        capture_report = dict(sample.metadata.get("capture_report", {}))
        rgb_ready = bool(np.asarray(sample.rgb).size > 0)
        depth_ready = bool(np.asarray(sample.depth).size > 0)
        nonempty_ready = bool(rgb_ready or depth_ready)
        pose_ready = bool(any(abs(float(v)) > 0.0 for v in sample.camera_pose_xyz) or float(sample.sim_time_s) != 0.0)
        if rgb_ready:
            self.tracker.succeed_phase("first_rgb_frame_ready", context={"timestamp": sample.sim_time_s, **capture_report})
        else:
            self.tracker.skip_phase("first_rgb_frame_ready", reason="RGB frame unavailable", context=capture_report)
        if depth_ready:
            self.tracker.succeed_phase("first_depth_frame_ready", context={"timestamp": sample.sim_time_s, **capture_report})
        else:
            self.tracker.skip_phase("first_depth_frame_ready", reason="Depth frame unavailable", context=capture_report)
        if nonempty_ready:
            self.tracker.succeed_phase(
                "first_nonempty_frame_ready",
                context={"rgb_ready": rgb_ready, "depth_ready": depth_ready, **capture_report},
            )
        else:
            self.tracker.skip_phase("first_nonempty_frame_ready", reason="RGB/depth tensors were empty", context=capture_report)
        pose_context = {
            "camera_pose_xyz": list(sample.camera_pose_xyz),
            "robot_pose_xyz": list(sample.robot_pose_xyz),
            "sim_time_s": float(sample.sim_time_s),
        }
        if pose_ready:
            self.tracker.succeed_phase("first_pose_ready", context=pose_context)
        else:
            self.tracker.skip_phase("first_pose_ready", reason="Pose metadata remained at defaults", context=pose_context)
        self.tracker.write_json_artifact(
            "first_frame_report",
            filename="first_frame_report.json",
            payload={
                "frame_id": sample.frame_id,
                "camera_pose_xyz": list(sample.camera_pose_xyz),
                "robot_pose_xyz": list(sample.robot_pose_xyz),
                "sim_time_s": float(sample.sim_time_s),
                "rgb_ready": rgb_ready,
                "depth_ready": depth_ready,
                "capture_report": capture_report,
            },
        )
        setattr(sample, "_pose_ready", pose_ready)
        setattr(sample, "_rgb_ready", rgb_ready)
        setattr(sample, "_depth_ready", depth_ready)
        return sample

    def _wait_for_first_sample(self, frame_source):
        for phase_name in ("first_rgb_frame_ready", "first_depth_frame_ready", "first_nonempty_frame_ready", "first_pose_ready"):
            self.tracker.start_phase(phase_name)
        deadline = self.time_fn() + float(self._profile_value("first_frame_timeout_sec"))
        last_notice = ""
        while self.time_fn() < deadline:
            sample = frame_source.read()
            if sample is not None:
                return sample
            report = frame_source.report()
            last_notice = str(getattr(report, "notice", ""))
            time.sleep(0.1)
        for phase_name, message in (
            ("first_rgb_frame_ready", "timed out waiting for first frame"),
            ("first_depth_frame_ready", "timed out waiting for first frame"),
            ("first_nonempty_frame_ready", "timed out waiting for a nonempty RGB/depth frame"),
            ("first_pose_ready", "timed out waiting for pose metadata"),
        ):
            combined_message = message if last_notice == "" else f"{message}; report={last_notice}"
            self.tracker.fail_phase(phase_name, message=combined_message, timeout=True)
        raise TimeoutError(last_notice or "Timed out waiting for live RGB/depth frame.")

    def _process_sample(self, sample) -> SmokeResultSummary:
        self.tracker.start_phase("observation_batch_processed")
        self.tracker.start_phase("perception_ingress_ready")
        self.tracker.start_phase("memory_updated")
        self.tracker.start_phase("memory_ingress_ready")
        memory_service = SmokeMemoryService()
        supervisor = self._supervisor_factory(memory_service=memory_service)
        batch = frame_sample_to_batch(sample)
        detection_attempted = batch.rgb_image is not None and batch.depth_image_m is not None
        enriched = supervisor.process_frame(batch, publish=False)
        observation_count = len(enriched.observations)
        detection_state = "detections produced" if observation_count > 0 else "frame received but no detections"
        self.tracker.succeed_phase(
            "observation_batch_processed",
            context={
                "observation_count": observation_count,
                "speaker_event_count": len(enriched.speaker_events),
                "ingress_state": detection_state,
            },
        )
        self.tracker.succeed_phase(
            "perception_ingress_ready",
            context={
                "detection_attempted": bool(detection_attempted),
                "detections_nonempty": bool(observation_count > 0),
            },
        )
        memory_ingress_ready = True
        if observation_count > 0:
            memory_service.update_from_observation(enriched.observations[0])
        memory_update_called = bool(memory_service.update_call_count > 0)
        memory_context = {
            "observe_call_count": memory_service.observe_call_count,
            "observe_item_count": memory_service.observe_item_count,
            "update_call_count": memory_service.update_call_count,
        }
        if memory_update_called:
            self.tracker.succeed_phase("memory_updated", context=memory_context)
        else:
            self.tracker.skip_phase("memory_updated", reason="No detections reached memory update path.", context=memory_context)
        self.tracker.succeed_phase("memory_ingress_ready", context=memory_context)

        result = aggregate_smoke_result(
            target_tier=self._smoke_target_tier(),
            frame_received=True,
            rgb_ready=bool(getattr(sample, "_rgb_ready", np.asarray(sample.rgb).size > 0)),
            depth_ready=bool(getattr(sample, "_depth_ready", np.asarray(sample.depth).size > 0)),
            pose_ready=bool(
                getattr(
                    sample,
                    "_pose_ready",
                    any(abs(float(v)) > 0.0 for v in sample.camera_pose_xyz) or float(sample.sim_time_s) != 0.0,
                )
            ),
            detection_attempted=bool(detection_attempted),
            detections_nonempty=bool(observation_count > 0),
            perception_ingress_ready=True,
            memory_ingress_ready=bool(memory_ingress_ready),
            memory_update_called=bool(memory_update_called),
        )
        self.tracker.set_smoke_result(result.as_dict())
        self.tracker.write_json_artifact("smoke_metrics", filename="smoke_metrics.json", payload=result.as_dict())
        return result

    def _mark_tier_phases(self, smoke_result: SmokeResultSummary) -> None:
        for phase_name, status in (
            ("sensor_smoke_pass", smoke_result.sensor_status),
            ("pipeline_smoke_pass", smoke_result.pipeline_status),
            ("memory_smoke_pass", smoke_result.memory_status),
        ):
            self.tracker.start_phase(phase_name)
            if status.passed:
                self.tracker.succeed_phase(phase_name, context={"tier": status.tier, "status": status.status, **status.details})
            else:
                self.tracker.skip_phase(phase_name, reason=status.reason or status.status, context={"tier": status.tier, **status.details})
        self.tracker.start_phase("full_smoke_pass")
        if smoke_result.sensor_status.passed and smoke_result.pipeline_status.passed and smoke_result.memory_status.passed:
            self.tracker.succeed_phase("full_smoke_pass", context=smoke_result.as_dict())
        else:
            self.tracker.skip_phase(
                "full_smoke_pass",
                reason=smoke_result.recommended_next_action or smoke_result.overall_status,
                context=smoke_result.as_dict(),
            )

    def _default_supervisor_factory(self, *, memory_service: MemoryService):
        return Supervisor(
            config=SupervisorConfig(memory_db_path=""),
            memory_service=memory_service,
        )

    def _standalone_available(self) -> bool:
        try:
            __import__("isaacsim")
        except Exception:
            return False
        return True

    def _editor_attach_available(self) -> bool:
        try:
            import omni.usd  # noqa: F401
        except Exception:
            return False
        return True

    def _emit_recommendations(
        self,
        *,
        compatibility_report: CompatibilityReport | None,
        failure_phase: str,
        smoke_result: SmokeResultSummary | None = None,
    ) -> None:
        if compatibility_report is None:
            return
        recommendations = build_recommendations(
            compatibility_report=compatibility_report,
            failure_phase=failure_phase,
            smoke_result=smoke_result,
        )
        if smoke_result is not None:
            updated = smoke_result.as_dict()
            updated["recommended_next_action"] = recommendations[0].action if recommendations else ""
            self.tracker.set_smoke_result(updated)
        for item in recommendations:
            self.tracker.add_recommendation_item(item.as_dict())

    def _recommend_failure_mode(self) -> None:
        recommended = recommend_mode_for_failure(
            selected_mode=str(self.tracker.diagnostics.launch_mode),
            failure_phase=str(self.tracker.diagnostics.failure_phase or self.tracker.diagnostics.current_phase),
        )
        if recommended != "":
            self.tracker.add_recommendation(f"retry with launch mode: {recommended}")

    def _print_compatibility(self, report: CompatibilityReport) -> None:
        for line in report.summary_lines():
            print(f"[LIVE_SMOKE][COMPAT] {line}")

    def _print_summary(self) -> None:
        diagnostics = self.tracker.diagnostics
        smoke_result = diagnostics.smoke_result
        sensor_status = smoke_result.get("sensor_status", {}).get("status", "-") if isinstance(smoke_result, dict) else "-"
        pipeline_status = smoke_result.get("pipeline_status", {}).get("status", "-") if isinstance(smoke_result, dict) else "-"
        memory_status = smoke_result.get("memory_status", {}).get("status", "-") if isinstance(smoke_result, dict) else "-"
        summary = diagnostics.summary or self.tracker.summary()
        print(
            "[LIVE_SMOKE] "
            f"status={diagnostics.status} "
            f"launch_mode={diagnostics.launch_mode} "
            f"profile={diagnostics.selected_profile or '-'} "
            f"target_tier={diagnostics.smoke_target_tier or '-'} "
            f"sensor_status={sensor_status} "
            f"pipeline_status={pipeline_status} "
            f"memory_status={memory_status} "
            f"failure_phase={diagnostics.failure_phase or '-'} "
            f"summary={summary}"
        )

    def _finalize_failure_with_recommendations(
        self,
        *,
        failure_phase: str,
        smoke_result: SmokeResultSummary | None = None,
    ) -> int:
        compatibility = self._compatibility_report
        if compatibility is not None:
            self._emit_recommendations(
                compatibility_report=compatibility,
                failure_phase=failure_phase,
                smoke_result=smoke_result,
            )
            self._print_compatibility(compatibility)
        else:
            self._recommend_failure_mode()
        self.tracker.finalize_failure()
        self._print_summary()
        return 1

    def _profile_value(self, field_name: str) -> int | float:
        profile = self._ensure_profile_selection(
            self._selected_launch.selected_mode if self._selected_launch is not None else str(getattr(self.args, "launch_mode", LAUNCH_MODE_STANDALONE))
        ).selected_profile
        cli_flag_map = {
            "render_warmup_updates": "--render-warmup-updates",
            "physics_warmup_steps": "--physics-warmup-steps",
            "stage_settle_updates": "--stage-settle-updates",
            "sensor_init_retries": "--sensor-init-retries",
            "sensor_init_retry_updates": "--sensor-init-retry-updates",
            "first_frame_timeout_sec": "--first-frame-timeout-sec",
        }
        flag_name = cli_flag_map.get(field_name, "")
        if flag_name != "" and self._explicit_arg_present(flag_name):
            return getattr(self.args, field_name)
        return getattr(profile, field_name)

    def _explicit_arg_present(self, *flags: str) -> bool:
        argv = list(getattr(self.args, "_argv", []))
        return any(flag in argv for flag in flags)

    def _requested_smoke_target_tier(self) -> str:
        return str(getattr(self.args, "smoke_target_tier", "sensor")).strip().lower() or "sensor"

    def _smoke_target_tier(self) -> str:
        if self._explicit_arg_present("--smoke-target-tier") or self._selected_profile is None:
            return self._requested_smoke_target_tier()
        return str(self._selected_profile.selected_profile.smoke_target_tier).strip().lower() or "sensor"


def _enabled_extensions() -> list[str]:
    try:
        import omni.kit.app
    except Exception:
        return []
    try:
        app = omni.kit.app.get_app()
    except Exception:
        return []
    if app is None:
        return []
    try:
        ext_mgr = app.get_extension_manager()
    except Exception:
        return []
    if ext_mgr is None:
        return []
    names: list[str] = []
    for ext_id in ext_mgr.get_enabled_extension_ids():
        try:
            ext_dict = ext_mgr.get_extension_dict(ext_id)
        except Exception:  # noqa: BLE001
            ext_dict = {}
        name = ext_dict.get("package", {}).get("name") if isinstance(ext_dict, dict) else None
        names.append(str(name or ext_id))
    return sorted(set(names))


def _extension_package_present() -> bool:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = repo_root / "exts" / "isaac.aura.live_smoke" / "config" / "extension.toml"
    return manifest.exists()


def _discover_isaac_root() -> str:
    env_root = str(os.environ.get("ISAAC_SIM_ROOT", "")).strip()
    if env_root != "":
        return str(Path(env_root).expanduser())
    env_python = str(os.environ.get("ISAAC_SIM_PYTHON", "")).strip()
    if env_python != "":
        return str(Path(env_python).expanduser().parent)
    exe_path = Path(sys.executable).resolve()
    candidates = [exe_path.parent, exe_path.parent.parent, Path("/mnt/c/isaac-sim"), Path("C:/isaac-sim")]
    for candidate in candidates:
        if (candidate / "python.bat").exists() or (candidate / "isaac-sim.bat").exists():
            return str(candidate)
    return str(exe_path.parent)


def _discover_isaac_python() -> str:
    env_python = str(os.environ.get("ISAAC_SIM_PYTHON", "")).strip()
    if env_python != "":
        return str(Path(env_python).expanduser())
    root = Path(_discover_isaac_root())
    for candidate in (root / "python.bat", root / "kit" / "python.bat"):
        if candidate.exists():
            return str(candidate)
    return str(Path(sys.executable).resolve())


def _safe_pid() -> int:
    try:
        return int(os.getpid())
    except Exception:  # noqa: BLE001
        return -1


def _call_update(simulation_app) -> None:  # noqa: ANN001
    update = getattr(simulation_app, "update", None)
    if callable(update):
        update()


def write_wrapper_timeout_summary(
    *,
    output_path: str | Path,
    diagnostics_path: str | Path,
    phase_name: str,
    timeout_sec: float,
    log_path: str,
    stderr_path: str,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(
            {
                "diagnostics_path": str(diagnostics_path),
                "timeout_phase": str(phase_name),
                "timeout_sec": float(timeout_sec),
                "stdout_log": str(log_path),
                "stderr_log": str(stderr_path),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
