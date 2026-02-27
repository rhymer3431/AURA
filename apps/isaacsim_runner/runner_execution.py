from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from apps.isaacsim_runner.runner_bridges import (
    NavigateCommandBridge,
    _run_mock_loop,
    _setup_ros2_camera_graph,
    _setup_ros2_joint_and_tf_graph,
)
from apps.isaacsim_runner.runner_config import DEFAULT_G1_START_Z, _prepare_internal_ros2_environment
from apps.isaacsim_runner.runner_control import _apply_g1_pd_gains, _log_robot_dof_snapshot
from apps.isaacsim_runner.runner_stage import (
    StageLayoutConfig,
    _apply_stage_layout,
    _parse_stage_reference_specs,
    _ensure_robot_dynamic_flags,
    _ensure_world_environment,
    _find_camera_prim_path,
    _find_motion_root_prim,
    _find_robot_prim_path,
    _read_base_pose,
    _rebase_world_anchor_articulation_root,
    _set_robot_start_height,
)


def _create_telemetry_logger(jsonl_logger_cls) -> Optional[object]:
    if jsonl_logger_cls is None:
        return None
    try:
        return jsonl_logger_cls(
            phase=os.environ.get("AURA_TELEMETRY_PHASE", "standing"),
            component="isaac_runner",
        )
    except Exception as exc:
        logging.warning("Failed to initialize runtime telemetry logger: %s", exc)
        return None


def _build_stage_layout_config(args: argparse.Namespace) -> StageLayoutConfig:
    environment_prim = str(getattr(args, "stage_environment_prim", "/World/Environment"))
    if not environment_prim.startswith("/"):
        environment_prim = f"/{environment_prim}"

    object_root_default = f"{environment_prim.rstrip('/')}/Objects"
    object_root_prim = str(getattr(args, "stage_object_root_prim", object_root_default))
    if not object_root_prim.startswith("/"):
        object_root_prim = f"/{object_root_prim}"

    environment_refs = _parse_stage_reference_specs(
        specs=getattr(args, "environment_ref", []) or [],
        default_parent_prim=environment_prim,
        default_prefix="EnvRef",
        kind="environment",
    )
    object_refs = _parse_stage_reference_specs(
        specs=getattr(args, "object_ref", []) or [],
        default_parent_prim=object_root_prim,
        default_prefix="ObjRef",
        kind="object",
    )

    flat_grid_prim = getattr(args, "flat_grid_prim", None)
    flat_grid_prim = str(flat_grid_prim).strip() if flat_grid_prim is not None else ""
    if not flat_grid_prim:
        flat_grid_prim = None

    return StageLayoutConfig(
        world_prim_path=str(getattr(args, "stage_world_prim", "/World")),
        environment_prim_path=environment_prim,
        robots_prim_path=str(getattr(args, "stage_robots_prim", "/World/Robots")),
        physics_scene_prim_path=str(getattr(args, "stage_physics_scene_prim", "/World/PhysicsScene")),
        key_light_prim_path=str(getattr(args, "stage_key_light_prim", f"{environment_prim.rstrip('/')}/KeyLight")),
        key_light_intensity=float(getattr(args, "stage_key_light_intensity", 500.0)),
        key_light_angle=float(getattr(args, "stage_key_light_angle", 0.53)),
        enable_flat_grid=not bool(getattr(args, "disable_flat_grid", False)),
        flat_grid_prim_path=flat_grid_prim,
        environment_refs=environment_refs,
        object_refs=object_refs,
    )


def _run_native_isaac(
    args: argparse.Namespace,
    jsonl_logger_cls=None,
    perf_now: Optional[Callable[[], float]] = None,
) -> None:
    _prepare_internal_ros2_environment()
    navigate_bridge: Optional[NavigateCommandBridge] = None
    telemetry = _create_telemetry_logger(jsonl_logger_cls)
    now_perf = perf_now if perf_now is not None else time.perf_counter

    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception as exc:
        try:
            from omni.isaac.kit import SimulationApp  # type: ignore
        except Exception:
            logging.warning("Isaac Sim modules unavailable. Falling back to mock mode: %s", exc)
            _run_mock_loop(args)
            return

    headless = not bool(args.gui)
    sim_app_config = {"headless": headless}
    if args.gui:
        # Use editor-style rendering defaults when GUI is requested.
        sim_app_config.update(
            {
                "hide_ui": False,
                "display_options": 3286,
                "window_width": 1920,
                "window_height": 1080,
            }
        )
    isaac_root = Path(os.environ.get("ISAAC_SIM_ROOT", r"C:\isaac-sim"))
    if args.gui:
        experience = isaac_root / "apps" / "isaacsim.exp.full.kit"
    else:
        experience = isaac_root / "apps" / "isaacsim.exp.base.python.kit"

    simulation_app = (
        SimulationApp(sim_app_config, experience=str(experience))
        if experience.exists()
        else SimulationApp(sim_app_config)
    )
    try:
        import omni.usd  # type: ignore
        from isaacsim.core.api import SimulationContext  # type: ignore
        from isaacsim.core.utils.extensions import enable_extension  # type: ignore
        from omni.isaac.core.articulations import Articulation  # type: ignore

        usd_path = Path(args.usd).resolve()
        if not usd_path.exists():
            raise FileNotFoundError(f"USD file not found: {usd_path}")

        enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()

        context = omni.usd.get_context()
        context.open_stage(str(usd_path))
        for _ in range(90):
            simulation_app.update()

        stage_layout = _build_stage_layout_config(args)
        stage_obj = context.get_stage()
        _ensure_world_environment(stage_obj, stage_layout)
        for _ in range(15):
            simulation_app.update()
        stage_obj = context.get_stage()

        if _apply_stage_layout(stage_obj, stage_layout):
            for _ in range(60):
                simulation_app.update()
            stage_obj = context.get_stage()
        _ensure_world_environment(stage_obj, stage_layout)
        for _ in range(15):
            simulation_app.update()
        stage_obj = context.get_stage()

        robot_prim_path = _find_robot_prim_path(stage_obj)
        motion_root_prim: Optional[str] = None
        camera_prim_path = _find_camera_prim_path(stage_obj)
        if robot_prim_path:
            robot_prim_path = _rebase_world_anchor_articulation_root(stage_obj, robot_prim_path)
            _ensure_robot_dynamic_flags(stage_obj, robot_prim_path)
            _set_robot_start_height(stage_obj, robot_prim_path, DEFAULT_G1_START_Z)
            motion_root_prim = _find_motion_root_prim(stage_obj, robot_prim_path)
            if args.enable_navigate_bridge:
                navigate_bridge = NavigateCommandBridge(args.namespace, motion_root_prim)
                navigate_bridge.start()
            else:
                logging.info(
                    "Navigate command bridge disabled by default (physics-safe). "
                    "Use --enable-navigate-bridge to force-enable root transform control."
                )

        logging.info("Loaded USD stage: %s", usd_path)
        logging.info("Isaac Sim display mode: %s", "GUI" if args.gui else "headless")
        if robot_prim_path:
            logging.info("Resolved articulation root path: %s", robot_prim_path)
        if telemetry is not None:
            telemetry.log(
                {
                    "event": "runner_startup",
                    "usd_path": str(usd_path),
                    "articulation_root_path": str(robot_prim_path or ""),
                    "motion_root_prim_path": str(motion_root_prim or ""),
                    "requested_rate_hz": float(args.rate_hz),
                    "flat_grid_enabled": bool(stage_layout.enable_flat_grid),
                    "environment_refs": [
                        {"usd_path": ref.usd_path, "prim_path": ref.prim_path}
                        for ref in stage_layout.environment_refs
                    ],
                    "object_refs": [
                        {"usd_path": ref.usd_path, "prim_path": ref.prim_path}
                        for ref in stage_layout.object_refs
                    ],
                }
            )
        if robot_prim_path:
            _setup_ros2_joint_and_tf_graph(args.namespace, robot_prim_path)
        else:
            logging.warning("Could not detect articulation robot prim. Joint command bridge is disabled.")
        enable_camera_bridge = bool(camera_prim_path) and (
            (not args.gui) or bool(args.enable_camera_bridge_in_gui)
        )
        if enable_camera_bridge:
            _setup_ros2_camera_graph(args.namespace, camera_prim_path)
        elif camera_prim_path and args.gui:
            logging.info(
                "GUI mode: ROS camera bridge is disabled by default to keep viewport interactive. "
                "Use --enable-camera-bridge-in-gui to force-enable it."
            )
        else:
            logging.warning("Could not find a Camera prim in the stage. RGB/Depth ROS topics are disabled.")

        if abs(float(args.rate_hz) - 50.0) > 1e-6:
            logging.warning(
                "Overriding --rate-hz=%.3f to fixed 50Hz for control-loop stability telemetry.",
                float(args.rate_hz),
            )
        control_hz = 50.0
        simulation_context = SimulationContext(
            physics_dt=1.0 / 200.0,
            rendering_dt=1.0 / control_hz,
            stage_units_in_meters=1.0,
        )
        cmd_dt = 1.0 / 50.0
        if telemetry is not None:
            telemetry.log(
                {
                    "event": "runner_timing_config",
                    "physics_dt": 1.0 / 200.0,
                    "control_dt": cmd_dt,
                    "render_dt": cmd_dt,
                }
            )
        simulation_context.initialize_physics()
        simulation_context.play()
        simulation_context.step(render=False)

        if robot_prim_path:
            try:
                robot = Articulation(robot_prim_path)
                robot.initialize()
                dof_snapshot = _log_robot_dof_snapshot(robot)
                pd_summary = _apply_g1_pd_gains(robot)
                if telemetry is not None:
                    telemetry.log(
                        {
                            "event": "dof_snapshot",
                            "articulation_root_path": str(robot_prim_path),
                            "num_dof": dof_snapshot.get("num_dof"),
                            "dof_names": dof_snapshot.get("dof_names"),
                        }
                    )
                    telemetry.log(
                        {
                            "event": "pd_gains_applied",
                            "matched_dof": pd_summary.get("matched_dof"),
                            "num_dof": pd_summary.get("num_dof"),
                            "remaining_dof_count": pd_summary.get("remaining_dof_count"),
                            "pd_groups": pd_summary.get("groups", []),
                        }
                    )
            except Exception as exc:
                logging.warning("Failed to initialize articulation for gain setup: %s", exc)

        loop_step_idx = 0
        loop_prev_start: Optional[float] = None
        loop_dt_window: list[float] = []
        base_prev_pose: Optional[Dict[str, float]] = None
        base_pose_path = motion_root_prim if motion_root_prim else robot_prim_path
        wall_start = now_perf()

        while simulation_app.is_running():
            t_loop_start = now_perf()
            if loop_prev_start is None:
                loop_dt = cmd_dt
            else:
                loop_dt = max(0.0, t_loop_start - loop_prev_start)
            loop_prev_start = t_loop_start

            if navigate_bridge is not None:
                navigate_bridge.spin_once()
                navigate_bridge.apply(stage_obj, cmd_dt)
            simulation_context.step(render=True)

            t_loop_end = now_perf()
            elapsed = max(0.0, t_loop_end - t_loop_start)
            overrun_ms = max(0.0, (elapsed - cmd_dt) * 1000.0)
            loop_dt_window.append(loop_dt)
            if len(loop_dt_window) > 400:
                loop_dt_window = loop_dt_window[-400:]

            publish_hz_window = None
            if loop_dt_window:
                avg_loop_dt = sum(loop_dt_window[-50:]) / len(loop_dt_window[-50:])
                if avg_loop_dt > 0.0:
                    publish_hz_window = 1.0 / avg_loop_dt

            loop_step_idx += 1
            rtf = None
            wall_elapsed = t_loop_end - wall_start
            if wall_elapsed > 1e-6:
                rtf = (loop_step_idx * cmd_dt) / wall_elapsed

            base_pose = _read_base_pose(stage_obj, base_pose_path) if base_pose_path else None
            base_roll = None
            base_pitch = None
            base_yaw = None
            base_height = None
            fall_flag = False
            slip_flag = False
            if base_pose is not None:
                base_roll = float(base_pose["roll"])
                base_pitch = float(base_pose["pitch"])
                base_yaw = float(base_pose["yaw"])
                base_height = float(base_pose["height"])
                fall_flag = bool(abs(base_roll) > 0.9 or abs(base_pitch) > 0.9 or base_height < 0.22)
                if base_prev_pose is not None and loop_dt > 1e-6:
                    dx = float(base_pose["x"] - base_prev_pose["x"])
                    dy = float(base_pose["y"] - base_prev_pose["y"])
                    base_speed_xy = math.sqrt(dx * dx + dy * dy) / loop_dt
                    cmd_speed = 0.0
                    if navigate_bridge is not None:
                        cmd_speed = math.sqrt(
                            float(navigate_bridge._vx) ** 2 + float(navigate_bridge._vy) ** 2
                        )
                    slip_flag = bool(
                        (cmd_speed < 0.05 and base_speed_xy > 0.20)
                        or (cmd_speed > 0.20 and base_speed_xy < 0.01)
                    )
                base_prev_pose = base_pose

            if telemetry is not None:
                rec: Dict[str, Any] = {
                    "event": "runner_loop",
                    "step_idx": loop_step_idx,
                    "loop_dt": float(loop_dt),
                    "loop_overrun_ms": float(overrun_ms),
                    "publish_hz_window": publish_hz_window,
                    "rtf": rtf,
                    "fall_flag": bool(fall_flag),
                    "slip_flag": bool(slip_flag),
                }
                if base_roll is not None:
                    rec["base_roll"] = base_roll
                if base_pitch is not None:
                    rec["base_pitch"] = base_pitch
                if base_yaw is not None:
                    rec["base_yaw"] = base_yaw
                if base_height is not None:
                    rec["base_height"] = base_height
                telemetry.log(rec)

        simulation_context.stop()
    except KeyboardInterrupt:
        logging.info("Isaac Sim runner interrupted by user.")
    finally:
        if navigate_bridge is not None:
            navigate_bridge.stop()
        if telemetry is not None:
            try:
                telemetry.flush()
                telemetry.close()
            except Exception:
                pass
        simulation_app.close()
