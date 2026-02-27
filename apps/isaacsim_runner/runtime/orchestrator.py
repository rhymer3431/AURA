from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from apps.isaacsim_runner.bridges.mock import run_mock_loop
from apps.isaacsim_runner.bridges.navigate import NavigateCommandBridge
from apps.isaacsim_runner.bridges.omnigraph import setup_camera_graph, setup_joint_and_tf_graph
from apps.isaacsim_runner.config.base import (
    CONTROL_HZ,
    DEFAULT_G1_START_Z,
    PHYSICS_HZ,
    prepare_internal_ros2_environment,
)
from apps.isaacsim_runner.config.stage import StageLayoutConfig, build_stage_layout_config
from apps.isaacsim_runner.control.g1 import apply_pd_gains, log_robot_dof_snapshot
from apps.isaacsim_runner.runtime.loop import run_simulation_loop
from apps.isaacsim_runner.stage.layout import apply_stage_layout, ensure_world_environment
from apps.isaacsim_runner.stage.prims import read_base_pose
from apps.isaacsim_runner.stage.robot import (
    ensure_robot_dynamic_flags,
    find_camera_prim_path,
    find_motion_root_prim,
    find_robot_prim_path,
    rebase_world_anchor_articulation_root,
    set_robot_start_height,
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


def _create_simulation_app(args: argparse.Namespace):
    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception as exc:
        try:
            from omni.isaac.kit import SimulationApp  # type: ignore
        except Exception:
            logging.warning("Isaac Sim modules unavailable. Falling back to mock mode: %s", exc)
            return None

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

    return (
        SimulationApp(sim_app_config, experience=str(experience))
        if experience.exists()
        else SimulationApp(sim_app_config)
    )


def _open_and_configure_stage(simulation_app, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    try:
        import omni.usd  # type: ignore
        from isaacsim.core.utils.extensions import enable_extension  # type: ignore
    except Exception as exc:
        logging.warning("Could not import Isaac stage/extension modules: %s", exc)
        return None

    usd_path = Path(args.usd).resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    context = omni.usd.get_context()
    context.open_stage(str(usd_path))
    for _ in range(90):
        simulation_app.update()

    stage_layout = build_stage_layout_config(args)
    stage_obj = context.get_stage()
    ensure_world_environment(stage_obj, stage_layout)
    for _ in range(15):
        simulation_app.update()
    stage_obj = context.get_stage()

    if apply_stage_layout(stage_obj, stage_layout):
        for _ in range(60):
            simulation_app.update()
        stage_obj = context.get_stage()
    ensure_world_environment(stage_obj, stage_layout)
    for _ in range(15):
        simulation_app.update()
    stage_obj = context.get_stage()

    return {
        "context": context,
        "usd_path": usd_path,
        "stage_layout": stage_layout,
        "stage_obj": stage_obj,
    }


def _initialize_robot(args: argparse.Namespace, stage_obj):
    navigate_bridge: Optional[NavigateCommandBridge] = None
    robot_prim_path = find_robot_prim_path(stage_obj)
    motion_root_prim: Optional[str] = None
    camera_prim_path = find_camera_prim_path(stage_obj)
    if robot_prim_path:
        robot_prim_path = rebase_world_anchor_articulation_root(stage_obj, robot_prim_path)
        ensure_robot_dynamic_flags(stage_obj, robot_prim_path)
        set_robot_start_height(stage_obj, robot_prim_path, DEFAULT_G1_START_Z)
        motion_root_prim = find_motion_root_prim(stage_obj, robot_prim_path)
        if args.enable_navigate_bridge:
            navigate_bridge = NavigateCommandBridge(args.namespace, motion_root_prim)
            navigate_bridge.start()
        else:
            logging.info(
                "Navigate command bridge disabled by default (physics-safe). "
                "Use --enable-navigate-bridge to force-enable root transform control."
            )

    return robot_prim_path, motion_root_prim, camera_prim_path, navigate_bridge


def _setup_bridges(
    args: argparse.Namespace,
    robot_prim_path: Optional[str],
    camera_prim_path: Optional[str],
) -> None:
    if robot_prim_path:
        setup_joint_and_tf_graph(args.namespace, robot_prim_path)
    else:
        logging.warning("Could not detect articulation robot prim. Joint command bridge is disabled.")
    enable_camera_bridge = bool(camera_prim_path) and ((not args.gui) or bool(args.enable_camera_bridge_in_gui))
    if enable_camera_bridge:
        setup_camera_graph(args.namespace, camera_prim_path)
    elif camera_prim_path and args.gui:
        logging.info(
            "GUI mode: ROS camera bridge is disabled by default to keep viewport interactive. "
            "Use --enable-camera-bridge-in-gui to force-enable it."
        )
    else:
        logging.warning("Could not find a Camera prim in the stage. RGB/Depth ROS topics are disabled.")


def _initialize_articulation(robot_prim_path: str, articulation_cls, telemetry) -> None:
    try:
        robot = articulation_cls(robot_prim_path)
        robot.initialize()
        dof_snapshot = log_robot_dof_snapshot(robot)
        pd_summary = apply_pd_gains(robot)
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


def _run_native_isaac(
    args: argparse.Namespace,
    jsonl_logger_cls=None,
    perf_now: Optional[Callable[[], float]] = None,
) -> None:
    prepare_internal_ros2_environment()
    navigate_bridge: Optional[NavigateCommandBridge] = None
    telemetry = _create_telemetry_logger(jsonl_logger_cls)
    now_perf = perf_now if perf_now is not None else time.perf_counter

    simulation_app = _create_simulation_app(args)
    if simulation_app is None:
        run_mock_loop(args)
        return

    try:
        try:
            from isaacsim.core.api import SimulationContext  # type: ignore
            from omni.isaac.core.articulations import Articulation  # type: ignore
        except Exception as exc:
            logging.warning("Could not import Isaac runtime modules: %s", exc)
            return

        stage_state = _open_and_configure_stage(simulation_app, args)
        if stage_state is None:
            return

        usd_path = stage_state["usd_path"]
        stage_layout: StageLayoutConfig = stage_state["stage_layout"]
        stage_obj = stage_state["stage_obj"]

        robot_prim_path, motion_root_prim, camera_prim_path, navigate_bridge = _initialize_robot(args, stage_obj)

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

        _setup_bridges(args, robot_prim_path, camera_prim_path)

        if abs(float(args.rate_hz) - CONTROL_HZ) > 1e-6:
            logging.warning(
                "Overriding --rate-hz=%.3f to fixed 50Hz for control-loop stability telemetry.",
                float(args.rate_hz),
            )
        control_hz = CONTROL_HZ
        simulation_context = SimulationContext(
            physics_dt=1.0 / PHYSICS_HZ,
            rendering_dt=1.0 / control_hz,
            stage_units_in_meters=1.0,
        )
        cmd_dt = 1.0 / CONTROL_HZ
        if telemetry is not None:
            telemetry.log(
                {
                    "event": "runner_timing_config",
                    "physics_dt": 1.0 / PHYSICS_HZ,
                    "control_dt": cmd_dt,
                    "render_dt": cmd_dt,
                }
            )
        simulation_context.initialize_physics()
        simulation_context.play()
        simulation_context.step(render=False)

        if robot_prim_path:
            _initialize_articulation(robot_prim_path, Articulation, telemetry)

        def _on_step(rec: Dict[str, Any]) -> None:
            if telemetry is not None:
                telemetry.log(rec)

        base_pose_path = motion_root_prim if motion_root_prim else robot_prim_path
        run_simulation_loop(
            simulation_app=simulation_app,
            simulation_context=simulation_context,
            stage_obj=stage_obj,
            cmd_dt=cmd_dt,
            now_perf=now_perf,
            navigate_bridge=navigate_bridge,
            base_pose_path=base_pose_path,
            read_base_pose_fn=read_base_pose,
            on_step=_on_step,
        )

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
