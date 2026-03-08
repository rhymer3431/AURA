from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "Detection2D": (".object_mapper", "Detection2D"),
    "DepthProjector": (".depth_projection", "DepthProjector"),
    "ObjectMapper": (".object_mapper", "ObjectMapper"),
    "ObservationFuser": (".observation_fuser", "ObservationFuser"),
    "PerceptionFrameResult": (".pipeline", "PerceptionFrameResult"),
    "PerceptionPipeline": (".pipeline", "PerceptionPipeline"),
    "PersonTrack": (".person_tracker", "PersonTrack"),
    "PersonTracker": (".person_tracker", "PersonTracker"),
    "ProjectedDetection": (".depth_projection", "ProjectedDetection"),
    "ReIdIdentity": (".reid_store", "ReIdIdentity"),
    "ReIdMatch": (".reid", "ReIdMatch"),
    "ReIdResolver": (".reid", "ReIdResolver"),
    "ReIdStore": (".reid_store", "ReIdStore"),
    "SpeakerEvent": (".speaker_events", "SpeakerEvent"),
    "build_viewer_overlay_payload": (".viewer_overlay", "build_viewer_overlay_payload"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
