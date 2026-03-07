from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from adapters.sensors.d455_mount import (
    DEFAULT_D455_ASSET_RELATIVE_PATH,
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
from runtime.isaac_launch_modes import (
    LAUNCH_MODE_ATTACH,
    LAUNCH_MODE_AUTO,
    LAUNCH_MODE_EXTENSION,
    LAUNCH_MODE_STANDALONE,
    LaunchModeAvailability,
    recommend_mode_for_failure,
    select_launch_mode,
)
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
            "assets_root_resolved": float(self.stage_ready_sec),
            "d455_asset_resolved": float(self.stage_ready_sec),
            "d455_prim_spawned": float(self.sensor_init_sec),
            "d455_depth_sensor_initialized": float(self.sensor_init_sec),
            "render_products_ready": float(self.sensor_init_sec),
            "first_rgb_frame_ready": float(self.first_frame_sec),
            "first_depth_frame_ready": float(self.first_frame_sec),
            "first_pose_ready": float(self.first_frame_sec),
            "observation_batch_processed": float(self.first_frame_sec),
            "memory_updated": float(self.first_frame_sec),
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

    def run_preflight(self) -> int:
        selection = self._record_launch_selection()
        self.tracker.start_phase("process_start", context={"mode": "preflight"})
        self.tracker.succeed_phase("process_start", context={"pid": _safe_pid()})
        try:
            self._record_env_resolution()
            self._record_extension_state(simulation_app=None)
            asset_resolution = self._resolve_d455_asset()
            self.tracker.finalize_success("preflight completed")
            self.tracker.write_json_artifact(
                "preflight_summary",
                filename="preflight_summary.json",
                payload={
                    "selected_launch_mode": selection.selected_mode,
                    "d455_asset_path": asset_resolution.asset_path,
                    "d455_asset_exists": asset_resolution.exists,
                    "recommended_mode": selection.recommended_mode,
                },
            )
            self._print_summary()
            return 0
        except Exception as exc:  # noqa: BLE001
            failure_phase = self.tracker.diagnostics.failure_phase or self.tracker.diagnostics.current_phase or "isaac_python_env_resolved"
            self.tracker.fail_phase(failure_phase, exc=exc)
            self._recommend_failure_mode()
            self.tracker.finalize_failure()
            self._print_summary()
            return 1

    def run_smoke(self) -> int:
        selection = self._record_launch_selection()
        if selection.selected_mode == LAUNCH_MODE_STANDALONE:
            return self._run_standalone_smoke(selection_reason=selection.reason)
        return self._run_attach_smoke(selection_reason=selection.reason)

    def _run_standalone_smoke(self, *, selection_reason: str) -> int:
        self.tracker.start_phase("process_start", context={"mode": "smoke"})
        self.tracker.succeed_phase("process_start", context={"pid": _safe_pid()})
        simulation_app = None
        try:
            self._record_env_resolution()
            simulation_app = self._create_simulation_app()
            stage, _ = self._prepare_stage(simulation_app)
            self.tracker.set_runtime_context(selected_launch_reason=selection_reason)
            return self._run_smoke_pipeline(simulation_app=simulation_app, stage=stage)
        except Exception as exc:  # noqa: BLE001
            failure_phase = self.tracker.diagnostics.failure_phase or self.tracker.diagnostics.current_phase or "simulation_app_created"
            self.tracker.fail_phase(failure_phase, exc=exc)
            self._recommend_failure_mode()
            self.tracker.finalize_failure()
            self._print_summary()
            return 1
        finally:
            if simulation_app is not None:
                close = getattr(simulation_app, "close", None)
                if callable(close):
                    close()

    def _run_attach_smoke(self, *, selection_reason: str) -> int:
        self.tracker.start_phase("process_start", context={"mode": "smoke_attach"})
        self.tracker.succeed_phase("process_start", context={"pid": _safe_pid()})
        try:
            self._record_env_resolution()
            stage, simulation_app = self._resolve_editor_stage()
            self.tracker.set_runtime_context(selected_launch_reason=selection_reason)
            return self._run_smoke_pipeline(simulation_app=simulation_app, stage=stage)
        except Exception as exc:  # noqa: BLE001
            failure_phase = self.tracker.diagnostics.failure_phase or self.tracker.diagnostics.current_phase or "stage_ready"
            self.tracker.fail_phase(failure_phase, exc=exc)
            self._recommend_failure_mode()
            self.tracker.finalize_failure()
            self._print_summary()
            return 1

    def _run_smoke_pipeline(self, *, simulation_app, stage) -> int:  # noqa: ANN001
        asset_resolution = self._resolve_d455_asset()
        mount_report = self._mount_d455(stage, asset_resolution.asset_path)
        sensor = self._initialize_sensor(simulation_app, stage)
        self.tracker.start_phase("render_products_ready")
        render_context = sensor.diagnostics_snapshot()
        self.tracker.write_json_artifact("sensor_diagnostics", filename="sensor_diagnostics.json", payload=render_context)
        if render_context.get("render_product_paths"):
            self.tracker.succeed_phase("render_products_ready", context=render_context)
        else:
            self.tracker.fail_phase(
                "render_products_ready",
                message="Camera initialized but render product paths were not discovered.",
                context=render_context,
            )
            self.tracker.add_recommendation("If render products stay empty in standalone headless mode, retry full_app_attach or extension_mode.")
            self.tracker.finalize_failure()
            self._print_summary()
            return 1

        sample = self._capture_first_sample(simulation_app, stage, sensor)
        smoke_metrics = self._process_sample(sample)
        full_success = (
            bool(smoke_metrics["frame_received"])
            and bool(smoke_metrics["pose_ready"])
            and bool(smoke_metrics["observation_batch_processed"])
            and bool(smoke_metrics["memory_updated"])
        )
        summary = (
            "smoke_pass reached"
            if full_success
            else "frame ingress reached but perception->memory update is incomplete"
        )
        if full_success:
            self.tracker.start_phase("smoke_pass", context=smoke_metrics)
            self.tracker.succeed_phase("smoke_pass", context={"mount_report": mount_report.as_dict()})
            self.tracker.finalize_success(summary)
            self._print_summary()
            return 0
        self.tracker.start_phase("smoke_pass", context=smoke_metrics)
        self.tracker.fail_phase(
            "smoke_pass",
            message=summary,
            context={
                "mount_report": mount_report.as_dict(),
                "detected_objects": smoke_metrics["observation_count"],
            },
        )
        if smoke_metrics["observation_count"] == 0:
            self.tracker.add_recommendation("Frame ingress succeeded but no detections reached memory. Verify detector output or use synthetic parity fixtures.")
        self.tracker.finalize_failure(summary)
        self._print_summary()
        return 1

    def _record_launch_selection(self):
        availability = LaunchModeAvailability(
            standalone_available=self._standalone_available(),
            editor_available=self._editor_attach_available(),
        )
        selection = select_launch_mode(str(getattr(self.args, "launch_mode", LAUNCH_MODE_AUTO)), availability=availability)
        self.tracker.set_runtime_context(
            launch_mode=selection.selected_mode,
            selected_launch_reason=selection.reason,
            context={
                "requested_launch_mode": selection.requested_mode,
                "recommended_mode": selection.recommended_mode,
                "standalone_available": availability.standalone_available,
                "editor_available": availability.editor_available,
            },
        )
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
                "python_executable": str(sys.executable),
                "isaac_python": isaac_python,
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
        extensions = _enabled_extensions()
        if simulation_app is not None:
            update = getattr(simulation_app, "update", None)
            if callable(update):
                update()
        self.tracker.set_runtime_context(enabled_extensions=extensions)
        self.tracker.write_json_artifact("enabled_extensions", filename="enabled_extensions.json", payload={"enabled_extensions": extensions})
        self.tracker.succeed_phase("required_extensions_ready", context={"enabled_extensions": extensions})

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
            world = World(stage_units_in_meters=1.0, physics_dt=float(getattr(self.args, "physics_dt", 1.0 / 60.0)), rendering_dt=rendering_dt)
        env_reference = resolve_environment_reference(getattr(self.args, "scene_usd", None), str(getattr(self.args, "env_url", "")))
        spawn_environment(env_reference, str(getattr(self.args, "scene_prim_path", "/World/Environment")), tuple(getattr(self.args, "scene_translate", (0.0, 0.0, 0.0))))
        for _ in range(max(int(getattr(self.args, "startup_updates", 4)), 1)):
            update = getattr(simulation_app, "update", None)
            if callable(update):
                update()
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable after SimulationApp bootstrap.")
        prim_tree = dump_prim_tree(stage)
        self.tracker.write_artifact("stage_prim_tree", filename="stage_prim_tree.txt", payload=prim_tree)
        self.tracker.succeed_phase(
            "stage_ready",
            context={
                "env_reference": env_reference,
                "scene_prim_path": str(getattr(self.args, "scene_prim_path", "/World/Environment")),
            },
        )
        return stage, world

    def _resolve_editor_stage(self):
        self.tracker.start_phase("stage_ready")
        try:
            import omni.usd
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("full_app_attach requires running inside Isaac Sim/Kit with omni.usd available.") from exc
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage found. Start Isaac Sim Full App first, then retry attach smoke.")
        try:
            import omni.kit.app
        except Exception:  # noqa: BLE001
            simulation_app = SimpleNamespace(update=lambda: None)
        else:
            simulation_app = omni.kit.app.get_app() or SimpleNamespace(update=lambda: None)
        prim_tree = dump_prim_tree(stage)
        self.tracker.write_artifact("stage_prim_tree", filename="stage_prim_tree.txt", payload=prim_tree)
        self.tracker.succeed_phase("stage_ready", context={"attach_mode": True})
        self._record_extension_state(simulation_app)
        return stage, simulation_app

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
        mount_report = ensure_d455_mount(stage, asset_path=asset_path, prim_path=str(getattr(self.args, "d455_prim_path", DEFAULT_D455_MOUNT_PRIM_PATH)))
        self.tracker.write_json_artifact("d455_mount_report", filename="d455_mount_report.json", payload=mount_report.as_dict())
        if not mount_report.prim_exists:
            raise RuntimeError(f"D455 prim mount failed at {mount_report.prim_path}")
        self.tracker.succeed_phase("d455_prim_spawned", context=mount_report.as_dict())
        return mount_report

    def _initialize_sensor(self, simulation_app, stage):
        self.tracker.start_phase("d455_depth_sensor_initialized")
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
        ok, message = sensor.initialize(simulation_app, stage)
        sensor_report = sensor.diagnostics_snapshot()
        sensor_report["initialize_ok"] = bool(ok)
        sensor_report["initialize_message"] = str(message)
        self.tracker.write_json_artifact("sensor_init_report", filename="sensor_init_report.json", payload=sensor_report)
        if not ok:
            raise RuntimeError(message)
        self.tracker.succeed_phase("d455_depth_sensor_initialized", context=sensor_report)
        return sensor

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
        frame_source.start()
        sample = self._wait_for_first_sample(frame_source)
        capture_report = dict(sample.metadata.get("capture_report", {}))
        if np.asarray(sample.rgb).size > 0:
            self.tracker.succeed_phase("first_rgb_frame_ready", context={"timestamp": sample.sim_time_s, **capture_report})
        else:
            self.tracker.skip_phase("first_rgb_frame_ready", reason="RGB frame unavailable", context=capture_report)
        if np.asarray(sample.depth).size > 0:
            self.tracker.succeed_phase("first_depth_frame_ready", context={"timestamp": sample.sim_time_s, **capture_report})
        else:
            self.tracker.skip_phase("first_depth_frame_ready", reason="Depth frame unavailable", context=capture_report)
        pose_context = {
            "camera_pose_xyz": list(sample.camera_pose_xyz),
            "robot_pose_xyz": list(sample.robot_pose_xyz),
            "sim_time_s": float(sample.sim_time_s),
        }
        pose_ready = any(abs(float(v)) > 0.0 for v in sample.camera_pose_xyz) or float(sample.sim_time_s) != 0.0
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
                "capture_report": capture_report,
            },
        )
        setattr(sample, "_pose_ready", bool(pose_ready))
        return sample

    def _wait_for_first_sample(self, frame_source: IsaacLiveFrameSource):
        self.tracker.start_phase("first_rgb_frame_ready")
        self.tracker.start_phase("first_depth_frame_ready")
        self.tracker.start_phase("first_pose_ready")
        deadline = self.time_fn() + float(self.phase_timeouts.first_frame_sec)
        sample = None
        last_notice = ""
        while self.time_fn() < deadline:
            sample = frame_source.read()
            if sample is not None:
                return sample
            report = frame_source.report()
            last_notice = report.notice
            time.sleep(0.1)
        self.tracker.fail_phase(
            "first_rgb_frame_ready",
            message=last_notice or "timed out waiting for first frame",
            timeout=True,
        )
        self.tracker.fail_phase(
            "first_depth_frame_ready",
            message=last_notice or "timed out waiting for first frame",
            timeout=True,
        )
        self.tracker.fail_phase(
            "first_pose_ready",
            message=last_notice or "timed out waiting for pose metadata",
            timeout=True,
        )
        raise TimeoutError(last_notice or "Timed out waiting for live RGB/depth frame.")

    def _process_sample(self, sample):
        self.tracker.start_phase("observation_batch_processed")
        memory_service = SmokeMemoryService()
        supervisor = self._supervisor_factory(memory_service=memory_service)
        batch = frame_sample_to_batch(sample)
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
        self.tracker.start_phase("memory_updated")
        memory_updated = observation_count > 0 and memory_service.observe_item_count > 0
        if observation_count > 0:
            memory_service.update_from_observation(enriched.observations[0])
        if memory_updated or memory_service.update_call_count > 0:
            self.tracker.succeed_phase(
                "memory_updated",
                context={
                    "observe_call_count": memory_service.observe_call_count,
                    "observe_item_count": memory_service.observe_item_count,
                    "update_call_count": memory_service.update_call_count,
                },
            )
        else:
            self.tracker.skip_phase(
                "memory_updated",
                reason="No detections reached memory update path.",
                context={
                    "observe_call_count": memory_service.observe_call_count,
                    "observe_item_count": memory_service.observe_item_count,
                    "update_call_count": memory_service.update_call_count,
                },
            )
        result = {
            "frame_received": True,
            "pose_ready": bool(getattr(sample, "_pose_ready", False)),
            "observation_batch_processed": True,
            "observation_count": observation_count,
            "memory_updated": bool(memory_updated or memory_service.update_call_count > 0),
            "memory_observe_call_count": memory_service.observe_call_count,
            "memory_update_call_count": memory_service.update_call_count,
        }
        self.tracker.write_json_artifact("smoke_metrics", filename="smoke_metrics.json", payload=result)
        return result

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

    def _recommend_failure_mode(self) -> None:
        recommended = recommend_mode_for_failure(
            selected_mode=str(self.tracker.diagnostics.launch_mode),
            failure_phase=str(self.tracker.diagnostics.failure_phase or self.tracker.diagnostics.current_phase),
        )
        if recommended != "":
            self.tracker.add_recommendation(f"retry with launch mode: {recommended}")

    def _print_summary(self) -> None:
        diagnostics = self.tracker.diagnostics
        summary = diagnostics.summary or self.tracker.summary()
        print(
            "[LIVE_SMOKE] "
            f"status={diagnostics.status} "
            f"launch_mode={diagnostics.launch_mode} "
            f"failure_phase={diagnostics.failure_phase or '-'} "
            f"summary={summary}"
        )


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


def _discover_isaac_root() -> str:
    candidates = [
        str(Path(sys.executable).resolve().parent),
        str(Path(sys.executable).resolve().parent.parent),
    ]
    for candidate in candidates:
        if (Path(candidate) / "python.bat").exists() or (Path(candidate) / "isaac-sim.bat").exists():
            return str(Path(candidate))
    return str(Path(sys.executable).resolve().parent)


def _discover_isaac_python() -> str:
    root = Path(_discover_isaac_root())
    for candidate in (root / "python.bat", root / "kit" / "python.bat"):
        if candidate.exists():
            return str(candidate)
    return str(Path(sys.executable).resolve())


def _safe_pid() -> int:
    try:
        import os

        return int(os.getpid())
    except Exception:  # noqa: BLE001
        return -1


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
