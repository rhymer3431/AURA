"""HTTP backend adapter for navigation-owned System2."""

from __future__ import annotations

from systems.inference.api.runtime import InternVlaNavClient


class System2HttpBackend(InternVlaNavClient):
    """Compatibility shim for the InternVLA HTTP backend."""


__all__ = ["System2HttpBackend"]
