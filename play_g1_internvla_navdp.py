"""InternVLA -> NavDP entrypoint for the standalone G1 ONNX runner."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from g1_play.entrypoint import main


if __name__ == "__main__":
    if "--control_mode" not in sys.argv:
        sys.argv.extend(["--control_mode", "internvla_navdp"])
    main()
