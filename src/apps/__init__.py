from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "create_dual_server_app": ("apps.dual_server_app", "create_app"),
    "create_navdp_server_app": ("apps.navdp_server_app", "create_app"),
    "dual_server_main": ("apps.dual_server_app", "main"),
    "navdp_server_main": ("apps.navdp_server_app", "main"),
    "parse_dual_args": ("apps.dual_server_app", "parse_args"),
    "parse_navdp_args": ("apps.navdp_server_app", "parse_args"),
    "frame_bridge_main": ("apps.frame_bridge_app", "main"),
    "frame_bridge_editor_attach": ("apps.frame_bridge_editor_app", "attach_current_stage"),
    "editor_smoke_main": ("apps.editor_smoke_entry", "main"),
    "editor_smoke_run": ("apps.editor_smoke_entry", "run_editor_smoke"),
    "editor_depth_viewer_main": ("apps.editor_depth_viewer_entry", "main"),
    "editor_depth_viewer_run": ("apps.editor_depth_viewer_entry", "run_depth_viewer"),
    "live_smoke_main": ("apps.live_smoke_app", "main"),
    "local_stack_main": ("apps.local_stack_app", "main"),
    "memory_agent_main": ("apps.memory_agent_app", "main"),
    "create_webrtc_gateway_app": ("apps.webrtc_gateway_app", "create_app"),
    "parse_webrtc_gateway_args": ("apps.webrtc_gateway_app", "parse_args"),
    "webrtc_gateway_main": ("apps.webrtc_gateway_app", "main"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
