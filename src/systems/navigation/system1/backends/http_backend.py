"""HTTP backend adapter for navigation-owned System1."""

from __future__ import annotations

from systems.navigation.client import NavDpClient


class System1HttpBackend(NavDpClient):
    """Compatibility shim for the NavDP HTTP backend."""


__all__ = ["System1HttpBackend"]
