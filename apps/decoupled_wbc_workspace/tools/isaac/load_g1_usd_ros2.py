import argparse
import os
import platform
import time


def _print_ros_env():
    keys = [
        "RMW_IMPLEMENTATION",
        "ROS_DISTRO",
        "ROS_VERSION",
        "AMENT_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
    ]
    print("[Isaac ROS2] Environment snapshot:")
    for key in keys:
        val = os.environ.get(key, "")
        print(f"  {key}={val}")
    print(f"  platform={platform.platform()}")


def _configure_ros_env(rmw_implementation: str):
    if not os.environ.get("RMW_IMPLEMENTATION"):
        os.environ["RMW_IMPLEMENTATION"] = rmw_implementation
    _print_ros_env()


def _enable_single_extension(ext_name: str):
    errors = []
    try:
        from omni.isaac.core.utils.extensions import enable_extension

        enable_extension(ext_name)
        print(f"[Isaac ROS2] Enabled extension via omni helper: {ext_name}")
        return
    except Exception as exc:
        errors.append(f"enable_extension failed: {exc}")

    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    try:
        ext_manager.set_extension_enabled_immediate(ext_name, True)
        print(f"[Isaac ROS2] Enabled extension via extension manager: {ext_name}")
        return
    except Exception as exc:
        errors.append(f"set_extension_enabled_immediate failed: {exc}")

    raise RuntimeError(f"Failed to enable {ext_name}: {' | '.join(errors)}")


def _enable_ros2_bridge(preferred_ext_name: str):
    if preferred_ext_name == "auto":
        candidates = ["isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"]
    else:
        candidates = [preferred_ext_name]

    failures = []
    for name in candidates:
        try:
            _enable_single_extension(name)
            return name
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    raise RuntimeError("ROS2 bridge startup failed. Attempts: " + " || ".join(failures))


def main():
    parser = argparse.ArgumentParser(description="Load a USD stage and enable Isaac Sim ROS2 bridge.")
    parser.add_argument("--usd-path", required=True, help="Absolute path to USD file")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim in headless mode")
    parser.add_argument("--warmup-sec", type=float, default=1.0, help="Seconds to wait before main loop")
    parser.add_argument(
        "--ros2-bridge-ext",
        default="auto",
        help="ROS2 bridge extension name: auto|isaacsim.ros2.bridge|omni.isaac.ros2_bridge",
    )
    parser.add_argument(
        "--rmw-implementation",
        default="rmw_fastrtps_cpp",
        help="RMW implementation to set when RMW_IMPLEMENTATION is not defined",
    )
    args = parser.parse_args()

    _configure_ros_env(args.rmw_implementation)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        enabled_ext = _enable_ros2_bridge(args.ros2_bridge_ext)
        print(f"[Isaac ROS2] Bridge extension active: {enabled_ext}")

        import omni.usd

        ok = omni.usd.get_context().open_stage(args.usd_path)
        if not ok:
            raise RuntimeError(f"Failed to open USD stage: {args.usd_path}")

        if args.warmup_sec > 0:
            t_end = time.time() + args.warmup_sec
            while time.time() < t_end and simulation_app.is_running():
                simulation_app.update()

        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
