from __future__ import annotations

import argparse
import os

import numpy as np

from common.geometry import quat_wxyz_to_yaw, step_pose_towards_target, yaw_to_quat_wxyz
from common.scene import disable_rigid_bodies, get_single_pose, resolve_environment_reference
from locomotion.paths import repo_dir, resolve_default_robot_usd_path

from .planning import PlannerSession


def run(args: argparse.Namespace) -> int:
    from isaacsim import SimulationApp

    launch_config = {"headless": bool(args.headless)}
    if bool(args.headless):
        launch_config["disable_viewport_updates"] = True
    simulation_app = SimulationApp(launch_config=launch_config)

    try:
        import omni.timeline
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import XFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path

        world = World(stage_units_in_meters=1.0)
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("[G1_POINTGOAL] Isaac Sim assets root not found")
            return 2

        env_url = str(args.env_url).strip()
        if env_url != "":
            env_asset_path = resolve_environment_reference(env_url, assets_root_path)
            add_reference_to_stage(usd_path=env_asset_path, prim_path="/World/Environment")
            print(f"[G1_POINTGOAL] environment loaded: {env_asset_path}")
        else:
            world.scene.add_default_ground_plane(prim_path="/World/GroundPlane", name="GroundPlane")
            print("[G1_POINTGOAL] environment not specified; using default ground plane")

        add_reference_to_stage(usd_path=args.usd_path, prim_path="/World/G1")
        stage = omni.usd.get_context().get_stage()

        disabled_count = disable_rigid_bodies(stage, "/World/G1")
        print(f"[G1_POINTGOAL] disabled rigidBodyEnabled attrs: {disabled_count}")

        root_prim = XFormPrim("/World/G1", name="g1_root_navdp")
        world.scene.add(root_prim)

        planner_session = PlannerSession(args)
        for _ in range(int(args.startup_updates)):
            simulation_app.update()
        planner_session.initialize(simulation_app, stage)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(int(args.startup_updates)):
            simulation_app.update()

        last_trajectory_world = np.zeros((0, 3), dtype=np.float32)
        last_waypoint_idx = 0
        last_plan_version = -1
        last_distance_m = -1.0

        try:
            for step in range(int(args.max_steps)):
                simulation_app.update()

                update = planner_session.update(step)
                if update.plan_version > last_plan_version:
                    last_plan_version = int(update.plan_version)
                    last_trajectory_world = np.asarray(update.trajectory_world, dtype=np.float32).copy()
                    last_waypoint_idx = 0

                root_pos, root_quat = get_single_pose(root_prim)
                if planner_session.mode == "pointgoal":
                    assert planner_session.goal_world_xy is not None
                    last_distance_m = float(np.linalg.norm(planner_session.goal_world_xy - root_pos[:2]))
                    if last_distance_m <= float(args.goal_tolerance_m):
                        print(f"[G1_POINTGOAL] goal reached at step={step} dist={last_distance_m:.3f}m")
                        return 0
                else:
                    no_response_sec = planner_session.no_response_sec()
                    if update.stop:
                        print(f"[G1_DUAL] stop signaled at step={step} goal_v={update.goal_version}")
                        return 0
                    if no_response_sec > float(args.safety_timeout_sec):
                        print(
                            "[G1_DUAL] safety timeout triggered "
                            f"no_response_sec={no_response_sec:.2f} limit={float(args.safety_timeout_sec):.2f}"
                        )
                        return 1

                if last_trajectory_world.shape[0] == 0:
                    if step % int(args.log_interval) == 0:
                        if planner_session.mode == "pointgoal":
                            if update.stats.last_error != "":
                                print(f"[G1_POINTGOAL][step={step}] planner waiting: {update.stats.last_error}")
                        else:
                            error_note = f" last_error={update.stats.last_error}" if update.stats.last_error != "" else ""
                            print(
                                "[G1_DUAL]"
                                f"[step={step}] waiting goal_v={update.goal_version} "
                                f"traj_v={update.traj_version} stale_sec={update.stale_sec:.2f}"
                                f"{error_note}"
                            )
                    continue

                while (
                    last_waypoint_idx < last_trajectory_world.shape[0] - 1
                    and np.linalg.norm(last_trajectory_world[last_waypoint_idx, :2] - root_pos[:2])
                    <= float(args.waypoint_reached_m)
                ):
                    last_waypoint_idx += 1

                target_idx = min(last_waypoint_idx + int(args.lookahead_index), last_trajectory_world.shape[0] - 1)
                target_xy = last_trajectory_world[target_idx, :2]
                current_yaw = quat_wxyz_to_yaw(root_quat)
                next_pos, next_yaw = step_pose_towards_target(
                    current_pos_xyz=root_pos,
                    current_yaw=current_yaw,
                    target_xy=target_xy,
                    max_step_m=float(args.max_step_m),
                    max_yaw_step_rad=float(args.max_yaw_step_rad),
                )
                next_quat = yaw_to_quat_wxyz(next_yaw)
                root_prim.set_world_poses(
                    positions=np.asarray([next_pos], dtype=np.float32),
                    orientations=np.asarray([next_quat], dtype=np.float32),
                )

                if step % int(args.log_interval) == 0:
                    error_note = f" last_error={update.stats.last_error}" if update.stats.last_error != "" else ""
                    if planner_session.mode == "pointgoal":
                        local_goal = update.goal_local_xy if update.goal_local_xy is not None else np.zeros(2, dtype=np.float32)
                        print(
                            "[G1_POINTGOAL]"
                            f"[step={step}] goal_dist={last_distance_m:.3f}m "
                            f"goal_local=({local_goal[0]:.3f},{local_goal[1]:.3f}) "
                            f"plan_ok={update.stats.successful_calls} plan_fail={update.stats.failed_calls} "
                            f"plan_latency_ms={update.stats.latency_ms:.1f}{error_note}"
                        )
                    else:
                        print(
                            "[G1_DUAL]"
                            f"[step={step}] goal_v={update.goal_version} traj_v={update.traj_version} "
                            f"stale_sec={update.stale_sec:.2f} "
                            f"plan_ok={update.stats.successful_calls} plan_fail={update.stats.failed_calls} "
                            f"plan_latency_ms={update.stats.latency_ms:.1f}{error_note}"
                        )
        finally:
            planner_session.shutdown()
            timeline.stop()

        print(
            "[G1_POINTGOAL] done "
            f"mode={planner_session.mode} final_dist_m={last_distance_m:.3f} "
            f"plan_success={planner_session.stats.successful_calls} plan_fail={planner_session.stats.failed_calls} "
            f"last_plan_step={planner_session.stats.last_plan_step}"
        )
        return 1
    finally:
        simulation_app.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G1 D455 navigation in Isaac Sim (pointgoal or dual mode).")
    parser.add_argument("--usd-path", type=str, default=resolve_default_robot_usd_path(repo_dir()))
    parser.add_argument("--env-url", type=str, default="/Isaac/Environments/Simple_Warehouse/warehouse.usd")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8888")
    parser.add_argument("--planner-mode", type=str, choices=("pointgoal", "dual"), default="pointgoal")
    parser.add_argument("--dual-server-url", type=str, default="http://127.0.0.1:8890")
    parser.add_argument("--instruction", type=str, default="Navigate safely to the target and stop when complete.")
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.4)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--plan-interval-frames", type=int, default=3)
    parser.add_argument("--dual-request-gap-frames", type=int, default=3)
    parser.add_argument("--safety-timeout-sec", type=float, default=20.0)
    parser.add_argument("--s1-period-sec", type=float, default=0.2)
    parser.add_argument("--s2-period-sec", type=float, default=1.0)
    parser.add_argument("--goal-ttl-sec", type=float, default=3.0)
    parser.add_argument("--traj-ttl-sec", type=float, default=1.5)
    parser.add_argument("--traj-max-stale-sec", type=float, default=4.0)
    parser.add_argument("--max-step-m", type=float, default=0.08)
    parser.add_argument("--max-yaw-step-rad", type=float, default=0.08)
    parser.add_argument("--lookahead-index", type=int, default=3)
    parser.add_argument("--waypoint-reached-m", type=float, default=0.2)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--strict-d455", action="store_true")
    parser.add_argument("--force-runtime-camera", action="store_true")
    parser.add_argument("--use-trajectory-z", action="store_true")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=640)
    parser.add_argument("--depth-max-m", type=float, default=5.0)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--reset-timeout-sec", type=float, default=15.0)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--stop-threshold", type=float, default=-3.0)
    parser.add_argument("--startup-updates", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=30)
    return parser.parse_args(argv)


def main() -> int:
    import sys

    args = parse_args(sys.argv[1:])
    usd_path_norm = os.path.normpath(str(args.usd_path))
    if not os.path.exists(usd_path_norm):
        print(f"[G1_POINTGOAL] usd-path not found: {usd_path_norm}")
        return 2
    if str(args.planner_mode).lower() == "pointgoal":
        if args.goal_x is None or args.goal_y is None:
            print("[G1_POINTGOAL] --goal-x and --goal-y are required in planner-mode=pointgoal")
            return 2
    else:
        if str(args.instruction).strip() == "":
            print("[G1_DUAL] --instruction must be non-empty in planner-mode=dual")
            return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
