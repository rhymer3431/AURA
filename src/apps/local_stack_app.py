"""Deprecated local-stack shim retained pending decommission."""

from __future__ import annotations

from apps.deprecated.local_stack_app import main as _legacy_main
from apps.deprecated.local_stack_app import parse_args


def main(argv: list[str] | None = None) -> int:
    # Deprecated wrapper: keep the public import path stable while local stack is decommissioned.
    print(
        "[DEPRECATED][LOCAL_STACK] local stack is no longer a canonical runtime surface. "
        "Use run_aura_runtime.ps1 for the main runtime or run_memory_agent.ps1 --loopback for fast diagnostics."
    )
    return _legacy_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
