"""Runtime orchestration for the standalone G1 policy runner."""

from __future__ import annotations

import os

import numpy as np

from .command import CommandSource, ConsoleCmdVelController
from .controller import G1PolicyController, infer_policy_backend
from .paths import (
    repo_dir,
    resolve_default_policy_path,
    resolve_default_robot_usd_path,
    resolve_environment_reference,
    select_onnx_providers,
)


def _resolve_runtime_paths(args) -> tuple[str, str, str | None]:
    base_dir = repo_dir()
    policy_path = os.path.abspath(args.policy) if args.policy else resolve_default_policy_path(base_dir)
    robot_usd = os.path.abspath(args.robot_usd) if args.robot_usd else resolve_default_robot_usd_path(base_dir)
    env_reference = resolve_environment_reference(args.scene_usd, args.env_url)
    return policy_path, robot_usd, env_reference


def _validate_runtime_paths(
    policy_path: str,
    robot_usd: str,
    env_reference: str | None,
    scene_usd: str | None,
):
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if not os.path.isfile(robot_usd):
        raise FileNotFoundError(f"G1 USD not found: {robot_usd}")
    if env_reference and scene_usd and not os.path.isfile(env_reference):
        raise FileNotFoundError(f"Scene USD not found: {env_reference}")


def _print_launch_summary(
    args,
    policy_path: str,
    robot_usd: str,
    env_reference: str | None,
    backend_name: str,
    providers: list[str],
):
    print(f"[INFO] Policy: {policy_path}")
    print(f"[INFO] Policy backend: {backend_name}")
    print(f"[INFO] G1 USD: {robot_usd}")
    if env_reference:
        print(f"[INFO] Environment: {env_reference}")
    if providers:
        print(f"[INFO] ONNX providers: {providers}")
    print(f"[INFO] Physics dt: {args.physics_dt}")
    print(f"[INFO] Decimation: {args.decimation}")
    print(f"[INFO] Action scale: {args.action_scale}")


def _build_physics_step_callback(world, controller, command_source: CommandSource):
    state = {"initialized": False, "reset_needed": False, "step": 0}

    def on_physics_step(step_size: float):
        del step_size

        if not state["initialized"]:
            controller.initialize()
            state["initialized"] = True
            return

        if state["reset_needed"]:
            world.reset(True)
            controller.reset()
            state["reset_needed"] = False
            state["initialized"] = False
            state["step"] = 0
            return

        controller.forward(state["step"], command_source.command())
        state["step"] += 1

    return state, on_physics_step


def run(args, simulation_app, command_source: CommandSource | None = None):
    import omni.usd
    from isaacsim.core.api import World

    from .scene import spawn_environment

    policy_path, robot_usd, env_reference = _resolve_runtime_paths(args)
    _validate_runtime_paths(policy_path, robot_usd, env_reference, args.scene_usd)

    rendering_dt = args.rendering_dt if args.rendering_dt > 0.0 else args.physics_dt * args.decimation
    backend_name = infer_policy_backend(policy_path)
    providers = select_onnx_providers(args.onnx_device) if backend_name == "onnxruntime" else []
    _print_launch_summary(args, policy_path, robot_usd, env_reference, backend_name, providers)

    if command_source is None:
        command_source = ConsoleCmdVelController(timeout=args.cmd_vel_timeout)
    requires_render = bool(getattr(command_source, "requires_render", False))
    render_world = (not args.headless) or requires_render

    shutdown_reason = ""
    controller = None

    try:
        world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt, rendering_dt=rendering_dt)
        spawn_environment(env_reference, args.scene_prim_path, tuple(args.scene_translate))

        controller = G1PolicyController(
            prim_path=args.robot_prim_path,
            usd_path=robot_usd,
            policy_path=policy_path,
            position=np.asarray(args.robot_position, dtype=np.float32),
            providers=providers,
            device_preference=args.onnx_device,
            decimation=args.decimation,
            action_scale=args.action_scale,
        )

        stage = omni.usd.get_context().get_stage()
        command_source.initialize(simulation_app, stage, controller)
        if isinstance(command_source, ConsoleCmdVelController):
            command_source.print_help()

        state, on_physics_step = _build_physics_step_callback(world, controller, command_source)

        world.reset()
        world.add_physics_callback("physics_step", callback_fn=on_physics_step)

        while simulation_app.is_running() and not command_source.quit_requested:
            if state["initialized"] and not state["reset_needed"]:
                command_source.update(state["step"])
            world.step(render=render_world)
            if world.is_stopped():
                state["reset_needed"] = True
            if args.max_steps > 0 and state["step"] >= args.max_steps:
                shutdown_reason = f"max_steps reached: step={state['step']} limit={args.max_steps}"
                break
        if shutdown_reason == "":
            if getattr(command_source, "quit_requested", False):
                shutdown_reason = str(getattr(command_source, "shutdown_reason", "")).strip() or "command source requested exit"
            elif not simulation_app.is_running():
                shutdown_reason = "simulation app is no longer running"
            else:
                shutdown_reason = "runtime loop exited"
        print(f"[INFO] Shutdown reason: {shutdown_reason}")
    finally:
        if controller is not None:
            controller.close()
        command_source.shutdown()

    return int(getattr(command_source, "exit_code", 0))
