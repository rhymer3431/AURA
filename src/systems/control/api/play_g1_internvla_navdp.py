"""InternVLA -> NavDP entrypoint for the standalone G1 runner."""

from __future__ import annotations

from importlib import import_module
import sys


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if "--control_mode" not in args:
        args.extend(["--control_mode", "internvla_navdp"])
    sys.argv = [sys.argv[0], *args]
    import_module("simulation.api.entrypoint").main()


if __name__ == "__main__":
    main()
