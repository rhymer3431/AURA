from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.contracts import Detection2D3D, Plan, pose_to_dict
from modules.exploration import ExplorationBehavior
from modules.manipulation_groot_trt import GrootManipulator
from modules.memory import SceneMemory
from modules.nav2_client import Nav2Client
from modules.perception_yoloe_trt import YOLOEPerception
from modules.planner_client import PlannerClient
from modules.slam_monitor import SLAMMonitor
from modules.task_executor import TaskExecutor
from modules.vram_guard import VRAMGuard


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        # JSON is valid YAML. The default config file follows JSON-compatible YAML.
        return json.loads(text)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [agent_runtime] %(message)s",
    )


async def _async_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def _build_world_state(memory: SceneMemory, slam: SLAMMonitor) -> Dict[str, Any]:
    pose = slam.latest_pose
    return {
        "robot_pose": pose_to_dict(pose),
        "c_loc": slam.c_loc,
        "slam_mode": slam.mode,
        "memory": memory.summary(max_objects=8),
    }


def _extract_targets(plan: Plan) -> List[str]:
    targets: List[str] = []
    for skill in plan.skills:
        obj = skill.args.get("object")
        if isinstance(obj, str) and obj.strip():
            targets.append(obj.strip().lower())
    return targets


async def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)
    runtime_cfg = cfg.get("runtime", {})
    _configure_logging(runtime_cfg.get("log_level", "INFO"))

    logging.info("Starting agent runtime with config: %s", config_path)
    memory = SceneMemory(cfg.get("memory", {}))
    slam = SLAMMonitor(cfg.get("slam", {}))
    nav = Nav2Client(cfg.get("nav2", {}), get_confidence=lambda: slam.c_loc)
    exploration = ExplorationBehavior(cfg.get("exploration", {}))
    manip = GrootManipulator(cfg.get("manipulation", {}))
    planner = PlannerClient(cfg.get("planner", {}))
    executor = TaskExecutor(memory, nav, manip, exploration, slam)
    slam.register_mode_callback(executor.on_slam_mode_changed)

    async def on_detections(dets: List[Detection2D3D]) -> None:
        memory.update_from_detection(dets, slam.latest_pose)

    perception = YOLOEPerception(cfg.get("perception", {}), on_detections=on_detections)
    vram_guard = VRAMGuard(cfg.get("vram_guard", {}))

    async def on_vram(level: int, free_mb: int, used_mb: int) -> None:
        planner.set_degrade_level(level if level >= 1 else 0)
        perception.set_degrade_level(level if level >= 2 else 0)
        if level >= 3:
            logging.warning(
                "VRAM degrade level 3: TODO reduce Isaac Sim render load (RGB/Depth only, no extra products)."
            )

    vram_guard.register_callback(on_vram)

    await perception.warmup()
    await manip.warmup()
    await slam.start()
    await perception.start()
    await vram_guard.start()

    # Start pose should be anchored once localization stream is available.
    memory.set_start_pose(slam.latest_pose)

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        # Windows default event loop may not support add_signal_handler.
        pass

    async def process_command(command: str) -> None:
        command = command.strip()
        if not command:
            return
        world_state = _build_world_state(memory, slam)
        plan = await planner.create_plan(command, world_state)
        memory.set_task_focus(_extract_targets(plan))
        logging.info("Plan notes: %s", plan.notes)
        await executor.execute_plan(plan)

    try:
        startup_command = args.command or runtime_cfg.get("startup_command", "")
        if startup_command:
            await process_command(startup_command)
            if runtime_cfg.get("exit_after_startup_command", False):
                stop_event.set()

        interactive = bool(runtime_cfg.get("interactive", True)) and not args.no_interactive
        while not stop_event.is_set() and interactive:
            try:
                user_command = await _async_input("agent> ")
            except EOFError:
                break
            if user_command.strip().lower() in {"quit", "exit"}:
                break
            await process_command(user_command)
    finally:
        await perception.stop()
        await slam.stop()
        await vram_guard.stop()
        await manip.stop()
        logging.info("Agent runtime stopped.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 agent runtime (mock-first, ROS2-ready).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to runtime config.yaml",
    )
    parser.add_argument("--command", type=str, default="", help="One-shot natural language command")
    parser.add_argument("--no-interactive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
