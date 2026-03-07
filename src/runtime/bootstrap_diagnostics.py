from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_LIVE_SMOKE_PHASES = (
    "process_start",
    "isaac_python_env_resolved",
    "simulation_app_created",
    "required_extensions_ready",
    "stage_ready",
    "assets_root_resolved",
    "d455_asset_resolved",
    "d455_prim_spawned",
    "d455_depth_sensor_initialized",
    "render_products_ready",
    "first_rgb_frame_ready",
    "first_depth_frame_ready",
    "first_pose_ready",
    "observation_batch_processed",
    "memory_updated",
    "smoke_pass",
)


@dataclass
class BootstrapPhaseRecord:
    name: str
    timeout_sec: float = 0.0
    status: str = "pending"
    success: bool = False
    started_at_s: float | None = None
    finished_at_s: float | None = None
    elapsed_ms: float = 0.0
    exception_class: str = ""
    exception_message: str = ""
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class BootstrapDiagnostics:
    launch_mode: str
    frame_source: str
    headless: bool
    diagnostics_path: str
    artifacts_dir: str
    script_path: str = ""
    isaac_root: str = ""
    isaac_python: str = ""
    python_executable: str = ""
    selected_launch_reason: str = ""
    status: str = "pending"
    current_phase: str = ""
    failure_phase: str = ""
    summary: str = ""
    created_at_s: float = field(default_factory=time.time)
    updated_at_s: float = field(default_factory=time.time)
    cli_args: list[str] = field(default_factory=list)
    enabled_extensions: list[str] = field(default_factory=list)
    phase_timeouts: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    phases: list[BootstrapPhaseRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phases"] = [asdict(phase) for phase in self.phases]
        return payload


class BootstrapPhaseTracker:
    def __init__(
        self,
        *,
        diagnostics_path: str | Path,
        artifact_dir: str | Path,
        launch_mode: str,
        frame_source: str,
        headless: bool,
        cli_args: list[str] | None = None,
        phase_timeouts: dict[str, float] | None = None,
    ) -> None:
        self._diagnostics_path = Path(diagnostics_path)
        self._artifacts_dir = Path(artifact_dir)
        self._diagnostics = BootstrapDiagnostics(
            launch_mode=str(launch_mode),
            frame_source=str(frame_source),
            headless=bool(headless),
            diagnostics_path=str(self._diagnostics_path),
            artifacts_dir=str(self._artifacts_dir),
            cli_args=list(cli_args or []),
            phase_timeouts={key: float(value) for key, value in dict(phase_timeouts or {}).items()},
            phases=[
                BootstrapPhaseRecord(name=phase_name, timeout_sec=float(dict(phase_timeouts or {}).get(phase_name, 0.0)))
                for phase_name in DEFAULT_LIVE_SMOKE_PHASES
            ],
        )
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush()

    @property
    def diagnostics(self) -> BootstrapDiagnostics:
        return self._diagnostics

    def set_runtime_context(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            if key == "enabled_extensions" and isinstance(value, list):
                self._diagnostics.enabled_extensions = [str(item) for item in value]
            elif hasattr(self._diagnostics, key):
                setattr(self._diagnostics, key, value)
            else:
                self._diagnostics.context[str(key)] = value
        self.flush()

    def add_recommendation(self, message: str) -> None:
        text = str(message).strip()
        if text == "" or text in self._diagnostics.recommendations:
            return
        self._diagnostics.recommendations.append(text)
        self.flush()

    def add_artifact(self, name: str, path: str | Path) -> None:
        self._diagnostics.artifacts[str(name)] = str(path)
        self.flush()

    def write_artifact(self, name: str, *, filename: str, payload: str | bytes) -> Path:
        path = self._artifacts_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(str(payload), encoding="utf-8")
        self.add_artifact(name, path)
        return path

    def write_json_artifact(self, name: str, *, filename: str, payload: dict[str, Any]) -> Path:
        return self.write_artifact(name, filename=filename, payload=json.dumps(payload, ensure_ascii=True, indent=2))

    def start_phase(self, name: str, *, context: dict[str, Any] | None = None) -> None:
        phase = self._phase(name)
        phase.status = "running"
        phase.success = False
        phase.started_at_s = time.time()
        phase.finished_at_s = None
        phase.elapsed_ms = 0.0
        phase.exception_class = ""
        phase.exception_message = ""
        if context:
            phase.context.update(dict(context))
        self._diagnostics.current_phase = str(name)
        self._diagnostics.updated_at_s = time.time()
        self.flush()

    def succeed_phase(self, name: str, *, context: dict[str, Any] | None = None) -> None:
        phase = self._phase(name)
        phase.finished_at_s = time.time()
        phase.elapsed_ms = self._elapsed_ms(phase)
        phase.status = "succeeded"
        phase.success = True
        if context:
            phase.context.update(dict(context))
        self._diagnostics.updated_at_s = time.time()
        self.flush()

    def skip_phase(self, name: str, *, reason: str, context: dict[str, Any] | None = None) -> None:
        phase = self._phase(name)
        if phase.started_at_s is None:
            phase.started_at_s = time.time()
        phase.finished_at_s = time.time()
        phase.elapsed_ms = self._elapsed_ms(phase)
        phase.status = "skipped"
        phase.success = False
        phase.exception_message = str(reason)
        if context:
            phase.context.update(dict(context))
        self._diagnostics.updated_at_s = time.time()
        self.flush()

    def fail_phase(
        self,
        name: str,
        *,
        exc: Exception | None = None,
        message: str = "",
        context: dict[str, Any] | None = None,
        timeout: bool = False,
    ) -> None:
        phase = self._phase(name)
        if phase.started_at_s is None:
            phase.started_at_s = time.time()
        phase.finished_at_s = time.time()
        phase.elapsed_ms = self._elapsed_ms(phase)
        phase.status = "timeout" if timeout else "failed"
        phase.success = False
        if exc is not None:
            phase.exception_class = type(exc).__name__
            phase.exception_message = str(exc)
        elif message != "":
            phase.exception_message = str(message)
        if context:
            phase.context.update(dict(context))
        self._diagnostics.status = "failed"
        self._diagnostics.failure_phase = str(name)
        self._diagnostics.current_phase = str(name)
        self._diagnostics.summary = self.summary()
        self._diagnostics.updated_at_s = time.time()
        self.flush()

    def finalize_success(self, summary: str = "") -> None:
        self._diagnostics.status = "succeeded"
        self._diagnostics.current_phase = ""
        self._diagnostics.summary = summary or self.summary()
        self._diagnostics.updated_at_s = time.time()
        self.flush()

    def finalize_failure(self, summary: str = "") -> None:
        if self._diagnostics.status != "failed":
            self._diagnostics.status = "failed"
        self._diagnostics.summary = summary or self.summary()
        self._diagnostics.updated_at_s = time.time()
        self.flush()

    def summary(self) -> str:
        for phase in self._diagnostics.phases:
            if phase.status in {"failed", "timeout"}:
                message = phase.exception_message or "phase failed"
                return f"failed at {phase.name}: {message}"
        for phase in reversed(self._diagnostics.phases):
            if phase.status == "succeeded":
                return f"last completed phase: {phase.name}"
            if phase.status == "running":
                return f"stalled during phase: {phase.name}"
        return "bootstrap not started"

    def flush(self) -> None:
        self._diagnostics_path.write_text(
            json.dumps(self._diagnostics.as_dict(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _phase(self, name: str) -> BootstrapPhaseRecord:
        normalized = str(name)
        for phase in self._diagnostics.phases:
            if phase.name == normalized:
                return phase
        phase = BootstrapPhaseRecord(
            name=normalized,
            timeout_sec=float(self._diagnostics.phase_timeouts.get(normalized, 0.0)),
        )
        self._diagnostics.phases.append(phase)
        return phase

    @staticmethod
    def _elapsed_ms(phase: BootstrapPhaseRecord) -> float:
        if phase.started_at_s is None:
            return 0.0
        return max((time.time() - float(phase.started_at_s)) * 1000.0, 0.0)
