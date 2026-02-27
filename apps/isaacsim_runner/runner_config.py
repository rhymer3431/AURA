from __future__ import annotations

import logging
import os
from pathlib import Path

DEFAULT_G1_START_Z = 0.43
DEFAULT_G1_GROUND_CLEARANCE_Z = 0.03
NAV_CMD_DEADBAND = 1e-4


def _default_usd_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "g1" / "g1_d455.usd"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [isaac_runner] %(message)s",
    )


def _prepare_internal_ros2_environment() -> None:
    isaac_root = Path(os.environ.get("ISAAC_SIM_ROOT", r"C:\isaac-sim"))
    ros2_lib = isaac_root / "exts" / "isaacsim.ros2.bridge" / "humble" / "lib"
    if not ros2_lib.exists():
        return

    os.environ.setdefault("ROS_DISTRO", "humble")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    path_sep = os.pathsep
    current_path = os.environ.get("PATH", "")
    entries = current_path.split(path_sep) if current_path else []
    ros2_lib_str = str(ros2_lib)
    if ros2_lib_str not in entries:
        os.environ["PATH"] = f"{current_path}{path_sep}{ros2_lib_str}" if current_path else ros2_lib_str
