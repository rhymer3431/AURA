"""Deprecated live-smoke shim retained pending decommission."""

from __future__ import annotations

from apps.deprecated.live_smoke_app import main as _legacy_main
from apps.deprecated.live_smoke_app import parse_args


def main(argv: list[str] | None = None) -> int:
    # Deprecated wrapper: keep legacy imports stable while diagnostics are moved out of the canonical runtime path.
    print(
        "[DEPRECATED][LIVE_SMOKE] live smoke is a decommissioning diagnostic path, not a canonical runtime surface."
    )
    return _legacy_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
