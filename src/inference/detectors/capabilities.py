from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BindingInfo:
    name: str
    index: int
    shape: tuple[int, ...]
    dtype: str
    mode: str


@dataclass
class DetectorRuntimeReport:
    backend_name: str
    engine_path: str = ""
    device: str = ""
    model_format: str = ""
    engine_exists: bool = False
    tensorrt_import_ok: bool = False
    deserialize_ok: bool = False
    serialization_mismatch: bool = False
    binding_metadata_ok: bool = False
    ready_for_inference: bool = False
    selected_backend: str = ""
    selected_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    inputs: list[BindingInfo] = field(default_factory=list)
    outputs: list[BindingInfo] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "backend_name": self.backend_name,
            "engine_path": self.engine_path,
            "device": self.device,
            "model_format": self.model_format,
            "engine_exists": self.engine_exists,
            "tensorrt_import_ok": self.tensorrt_import_ok,
            "deserialize_ok": self.deserialize_ok,
            "serialization_mismatch": self.serialization_mismatch,
            "binding_metadata_ok": self.binding_metadata_ok,
            "ready_for_inference": self.ready_for_inference,
            "selected_backend": self.selected_backend,
            "selected_reason": self.selected_reason,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "inputs": [binding.__dict__ for binding in self.inputs],
            "outputs": [binding.__dict__ for binding in self.outputs],
        }


@dataclass(frozen=True)
class DetectorSelection:
    backend: object
    report: DetectorRuntimeReport
