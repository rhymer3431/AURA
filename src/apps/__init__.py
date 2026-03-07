from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "create_dual_server_app": ("apps.legacy_http.dual_server_app", "create_app"),
    "create_navdp_server_app": ("apps.legacy_http.navdp_server_app", "create_app"),
    "dual_server_main": ("apps.legacy_http.dual_server_app", "main"),
    "navdp_server_main": ("apps.legacy_http.navdp_server_app", "main"),
    "parse_dual_args": ("apps.legacy_http.dual_server_app", "parse_args"),
    "parse_navdp_args": ("apps.legacy_http.navdp_server_app", "parse_args"),
    "isaac_bridge_main": ("apps.isaac_bridge_app", "main"),
    "local_stack_main": ("apps.local_stack_app", "main"),
    "memory_agent_main": ("apps.memory_agent_app", "main"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
