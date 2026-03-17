"""Deprecated compatibility entry for the canonical navigation runtime."""

from __future__ import annotations

from .navigation_runtime import NavigationRuntime, build_launch_config, main


class AuraRuntimeCommandSource(NavigationRuntime):
    """Deprecated alias kept for launcher and test compatibility."""


__all__ = ["AuraRuntimeCommandSource", "NavigationRuntime", "build_launch_config", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
