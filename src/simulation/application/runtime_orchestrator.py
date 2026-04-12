"""Runtime orchestration for the standalone G1 locomotion runner."""

from __future__ import annotations

import os

import numpy as np
from isaacsim.core.api import World

from simulation.domain.constants import DEFAULT_DECIMATION, DEFAULT_PHYSICS_DT
from simulation.infrastructure.paths import (
    repo_dir,
    resolve_default_policy_path,
    resolve_default_robot_usd_path,
    resolve_environment_reference,
    select_onnx_providers,
)
from simulation.infrastructure.policy_controller import G1PolicyController
from simulation.infrastructure.policy_session import infer_policy_backend
from simulation.infrastructure.scene import spawn_environment
from simulation.infrastructure.training_config import RuntimeTrainingConfig, load_runtime_training_config
from simulation.application.runtime_coordinator import NavigationRuntimeCoordinator
from systems.control.api.runtime_controller import InternVlaNavDpController
from systems.control.operator_input import ConsoleCmdVelController, KeyboardCmdVelController
from systems.perception.api.camera_api import RuntimeCameraPitchService


def _resolve_runtime_paths(args) -> tuple[str, str, str | None, str | None]:
    base_dir = repo_dir()
    policy_path = os.path.abspath(args.policy) if args.policy else resolve_default_policy_path(base_dir)
    robot_usd = os.path.abspath(args.robot_usd) if args.robot_usd else resolve_default_robot_usd_path(base_dir)
    env_reference = resolve_environment_reference(args.scene_usd, args.env_url)
    config_dir = os.path.abspath(args.config_dir) if args.config_dir else None
    return policy_path, robot_usd, env_reference, config_dir


def _validate_runtime_paths(
    policy_path: str,
    robot_usd: str,
    env_reference: str | None,
    scene_usd: str | None,
    config_dir: str | None,
):
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if not os.path.isfile(robot_usd):
        raise FileNotFoundError(f"G1 USD not found: {robot_usd}")
    if env_reference and scene_usd and not os.path.isfile(env_reference):
        raise FileNotFoundError(f"Scene USD not found: {env_reference}")
    if config_dir and not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Training config directory not found: {config_dir}")


def _validate_default_policy_device(args, policy_path: str) -> None:
    if str(getattr(args, "policy", "")).strip() != "":
        return
    if infer_policy_backend(policy_path) != "tensorrt":
        return
    if str(getattr(args, "onnx_device", "auto")).strip().lower() != "cpu":
        return
    raise RuntimeError(
        "Default locomotion policy uses a TensorRT G1 locomotion engine and requires CUDA/TensorRT. "
        "Use --onnx_device auto/cuda, or pass an explicit ONNX policy with --policy."
    )


def _resolve_runtime_defaults(args, training_config: RuntimeTrainingConfig | None) -> dict:
    physics_dt = (
        float(args.physics_dt)
        if args.physics_dt is not None
        else (
            float(training_config.physics_dt)
            if training_config is not None and training_config.physics_dt is not None
            else DEFAULT_PHYSICS_DT
        )
    )
    decimation = (
        int(args.decimation)
        if args.decimation is not None
        else (
            int(training_config.decimation)
            if training_config is not None and training_config.decimation is not None
            else DEFAULT_DECIMATION
        )
    )
    robot_position = (
        tuple(float(value) for value in args.robot_position)
        if args.robot_position is not None
        else (
            training_config.robot_position
            if training_config is not None and training_config.robot_position is not None
            else (0.0, 0.0, 0.8)
        )
    )
    action_scale = args.action_scale
    if action_scale is None and training_config is not None:
        action_scale = training_config.action_scale

    height_scan_size = tuple(args.height_scan_size) if args.height_scan_size else None
    if height_scan_size is None and training_config is not None:
        height_scan_size = training_config.height_scan_size

    height_scan_resolution = args.height_scan_resolution
    if height_scan_resolution is None and training_config is not None:
        height_scan_resolution = training_config.height_scan_resolution

    height_scan_enabled = None
    if training_config is not None:
        height_scan_enabled = training_config.height_scan_enabled
    if args.height_scan_size or args.height_scan_resolution is not None:
        height_scan_enabled = True

    return {
        "physics_dt": physics_dt,
        "decimation": decimation,
        "robot_position": robot_position,
        "action_scale": action_scale,
        "height_scan_size": height_scan_size,
        "height_scan_resolution": height_scan_resolution,
        "height_scan_enabled": height_scan_enabled,
        "default_joint_pos_patterns": (
            training_config.default_joint_pos_patterns if training_config is not None else None
        ),
        "stiffness_patterns": training_config.stiffness_patterns if training_config is not None else None,
        "damping_patterns": training_config.damping_patterns if training_config is not None else None,
        "solver_position_iterations": (
            training_config.solver_position_iterations if training_config is not None else None
        ),
        "solver_velocity_iterations": (
            training_config.solver_velocity_iterations if training_config is not None else None
        ),
    }


def _print_launch_summary(
    resolved: dict,
    training_config: RuntimeTrainingConfig | None,
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
    if training_config is not None:
        print(f"[INFO] Training config: {training_config.source_path}")
    if providers:
        print(f"[INFO] ONNX providers: {providers}")
    print(f"[INFO] Physics dt: {resolved['physics_dt']}")
    print(f"[INFO] Decimation: {resolved['decimation']}")


def _build_physics_step_callback(world: World, controller: G1PolicyController, operator_input, services=None):
    state = {"first_step": True, "reset_needed": False, "step": 0}
    services = list(services or [])

    def on_physics_step(step_size: float):
        del step_size

        if state["first_step"]:
            controller.initialize()
            if hasattr(operator_input, "bind_controller"):
                operator_input.bind_controller(controller)
            for service in services:
                if hasattr(service, "bind_controller"):
                    service.bind_controller(controller)
            state["first_step"] = False
            return

        if state["reset_needed"]:
            world.reset(True)
            controller.reset()
            if hasattr(operator_input, "reset"):
                operator_input.reset()
            for service in services:
                if hasattr(service, "reset"):
                    service.reset()
            state["reset_needed"] = False
            state["first_step"] = True
            state["step"] = 0
            return

        controller.forward(state["step"], operator_input.command())
        state["step"] += 1

    return state, on_physics_step


def _create_operator_input(args):
    if args.control_mode == "cmd_vel":
        operator_input = ConsoleCmdVelController(timeout=args.cmd_vel_timeout)
    elif args.control_mode == "keyboard":
        if args.headless:
            raise ValueError("Keyboard control requires GUI mode. Remove --headless.")
        operator_input = KeyboardCmdVelController(
            lin_speed=args.lin_speed,
            lat_speed=args.lat_speed,
            yaw_speed=args.yaw_speed,
            require_focus=args.require_keyboard_focus,
        )
    elif args.control_mode == "internvla_navdp":
        operator_input = InternVlaNavDpController(args)
    else:
        raise ValueError(f"Unsupported control mode: {args.control_mode}")
    operator_input.print_help()
    return operator_input


def run(args, simulation_app):
    policy_path, robot_usd, env_reference, config_dir = _resolve_runtime_paths(args)
    _validate_runtime_paths(policy_path, robot_usd, env_reference, args.scene_usd, config_dir)
    _validate_default_policy_device(args, policy_path)

    training_config = load_runtime_training_config(config_dir) if config_dir else None
    resolved = _resolve_runtime_defaults(args, training_config)

    rendering_dt = (
        args.rendering_dt if args.rendering_dt > 0.0 else resolved["physics_dt"] * resolved["decimation"]
    )
    backend_name = infer_policy_backend(policy_path)
    providers = select_onnx_providers(args.onnx_device) if backend_name == "onnxruntime" else []
    _print_launch_summary(resolved, training_config, policy_path, robot_usd, env_reference, backend_name, providers)

    operator_input = None
    services = []
    try:
        world = World(stage_units_in_meters=1.0, physics_dt=resolved["physics_dt"], rendering_dt=rendering_dt)
        spawn_environment(env_reference, args.scene_prim_path, tuple(args.scene_translate))

        operator_input = _create_operator_input(args)
        if args.control_mode == "internvla_navdp":
            services.append(NavigationRuntimeCoordinator(args, control_handler=operator_input))
        if args.camera_api_port > 0 and args.control_mode != "internvla_navdp":
            services.append(RuntimeCameraPitchService(args))

        print(f"[INFO] Creating locomotion controller from: {policy_path}")
        controller = G1PolicyController(
            prim_path=args.robot_prim_path,
            usd_path=robot_usd,
            policy_path=policy_path,
            position=np.asarray(resolved["robot_position"], dtype=np.float32),
            decimation=resolved["decimation"],
            physics_dt=resolved["physics_dt"],
            providers=providers,
            device_preference=args.onnx_device,
            action_scale=resolved["action_scale"],
            height_scan_size=resolved["height_scan_size"],
            height_scan_resolution=resolved["height_scan_resolution"],
            height_scan_offset=args.height_scan_offset,
            default_joint_pos_patterns=resolved["default_joint_pos_patterns"],
            stiffness_patterns=resolved["stiffness_patterns"],
            damping_patterns=resolved["damping_patterns"],
            solver_position_iterations=resolved["solver_position_iterations"],
            solver_velocity_iterations=resolved["solver_velocity_iterations"],
            height_scan_enabled=resolved["height_scan_enabled"],
        )

        state, on_physics_step = _build_physics_step_callback(world, controller, operator_input, services=services)

        world.reset()
        world.add_physics_callback("physics_step", callback_fn=on_physics_step)

        while simulation_app.is_running() and not operator_input.quit_requested:
            if not state["first_step"] and not state["reset_needed"]:
                for service in services:
                    if hasattr(service, "step"):
                        service.step()
            should_render = (not args.headless) or bool(getattr(operator_input, "requires_render", False))
            world.step(render=should_render)
            if world.is_stopped():
                state["reset_needed"] = True
            if args.max_steps > 0 and state["step"] >= args.max_steps:
                break
    finally:
        if "controller" in locals() and hasattr(controller, "shutdown"):
            controller.shutdown()
        for service in services:
            if hasattr(service, "shutdown"):
                service.shutdown()
        if operator_input is not None:
            operator_input.shutdown()
