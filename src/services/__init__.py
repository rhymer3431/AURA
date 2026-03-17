from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "DualOrchestrator": ("services.dual_orchestrator", "DualOrchestrator"),
    "MemoryPolicyService": ("services.memory_policy_service", "MemoryPolicyService"),
    "MemoryService": ("services.memory_service", "MemoryService"),
    "NavDPInferenceService": ("services.navdp_inference_service", "NavDPInferenceService"),
    "TaskOrchestrator": ("services.task_orchestrator", "TaskOrchestrator"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
