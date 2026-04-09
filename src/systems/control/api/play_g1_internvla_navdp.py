"""InternVLA -> NavDP entrypoint for the standalone G1 runner."""

from __future__ import annotations

import sys

from systems.control.application.entrypoint import main


if __name__ == "__main__":
    if "--control_mode" not in sys.argv:
        sys.argv.extend(["--control_mode", "internvla_navdp"])
    main()
