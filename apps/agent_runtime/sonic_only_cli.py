from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.manipulation_groot_trt import GrootManipulator


@dataclass
class SonicVelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    style: str = "normal"
    duration_s: float = 2.5


LOCOMOTION_MAP = {
    "walk forward": SonicVelocityCommand(vx=0.25, vy=0.0, yaw_rate=0.0, style="normal", duration_s=2.5),
    "go forward": SonicVelocityCommand(vx=0.25, vy=0.0, yaw_rate=0.0, style="normal", duration_s=2.5),
    "walk backward": SonicVelocityCommand(vx=-0.18, vy=0.0, yaw_rate=0.0, style="normal", duration_s=2.5),
    "go back": SonicVelocityCommand(vx=-0.18, vy=0.0, yaw_rate=0.0, style="normal", duration_s=2.5),
    "run": SonicVelocityCommand(vx=0.8, vy=0.0, yaw_rate=0.0, style="run", duration_s=2.0),
    "turn left": SonicVelocityCommand(vx=0.0, vy=0.0, yaw_rate=0.45, style="normal", duration_s=1.8),
    "turn right": SonicVelocityCommand(vx=0.0, vy=0.0, yaw_rate=-0.45, style="normal", duration_s=1.8),
    "walk left": SonicVelocityCommand(vx=0.0, vy=0.2, yaw_rate=0.0, style="normal", duration_s=2.5),
    "go left": SonicVelocityCommand(vx=0.0, vy=0.2, yaw_rate=0.0, style="normal", duration_s=2.5),
    "walk right": SonicVelocityCommand(vx=0.0, vy=-0.2, yaw_rate=0.0, style="normal", duration_s=2.5),
    "go right": SonicVelocityCommand(vx=0.0, vy=-0.2, yaw_rate=0.0, style="normal", duration_s=2.5),
    "sneak": SonicVelocityCommand(vx=0.15, vy=0.0, yaw_rate=0.0, style="stealth", duration_s=3.0),
    "walk around": SonicVelocityCommand(vx=0.16, vy=0.0, yaw_rate=0.35, style="happy", duration_s=6.0),
    "stop": SonicVelocityCommand(vx=0.0, vy=0.0, yaw_rate=0.0, style="normal", duration_s=0.1),
}

KEYBOARD_TELEOP_MAP = {
    "w": SonicVelocityCommand(vx=0.25, vy=0.0, yaw_rate=0.0, style="normal", duration_s=0.12),
    "s": SonicVelocityCommand(vx=-0.18, vy=0.0, yaw_rate=0.0, style="normal", duration_s=0.12),
    "a": SonicVelocityCommand(vx=0.0, vy=0.2, yaw_rate=0.0, style="normal", duration_s=0.12),
    "d": SonicVelocityCommand(vx=0.0, vy=-0.2, yaw_rate=0.0, style="normal", duration_s=0.12),
    "q": SonicVelocityCommand(vx=0.0, vy=0.0, yaw_rate=0.45, style="normal", duration_s=0.12),
    "e": SonicVelocityCommand(vx=0.0, vy=0.0, yaw_rate=-0.45, style="normal", duration_s=0.12),
    "r": SonicVelocityCommand(vx=0.8, vy=0.0, yaw_rate=0.0, style="run", duration_s=0.12),
    "f": SonicVelocityCommand(vx=0.15, vy=0.0, yaw_rate=0.0, style="stealth", duration_s=0.12),
    " ": SonicVelocityCommand(vx=0.0, vy=0.0, yaw_rate=0.0, style="normal", duration_s=0.08),
}

KEYBOARD_TELEOP_IDLE_TIMEOUT_S = 0.35


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [sonic_only_cli] %(message)s",
    )


async def _async_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def _parse_locomotion_intent(text: str) -> Optional[SonicVelocityCommand]:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    for pattern, cmd in LOCOMOTION_MAP.items():
        if pattern in compact:
            return cmd
    return None


async def _run_locomotion(manip: GrootManipulator, cmd: SonicVelocityCommand, source: str) -> bool:
    manip.set_sonic_velocity(cmd)
    return await manip.execute_locomotion(
        linear_x=float(cmd.vx),
        linear_y=float(cmd.vy),
        angular_z=float(cmd.yaw_rate),
        duration_s=float(cmd.duration_s),
        source=source,
    )


async def _run_keyboard_teleop(manip: GrootManipulator) -> None:
    if sys.platform != "win32":
        logging.warning("Keyboard teleop is currently supported on Windows only.")
        return
    try:
        import msvcrt  # type: ignore
    except Exception as exc:
        logging.warning("Keyboard teleop is unavailable: %s", exc)
        return

    @dataclass
    class _TeleopState:
        command: SonicVelocityCommand
        updated_at: float

    def _read_key_blocking() -> str:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ext = msvcrt.getwch()
            arrow_map = {"H": "w", "P": "s", "K": "q", "M": "e"}
            return arrow_map.get(ext, "")
        return ch

    stop_cmd = KEYBOARD_TELEOP_MAP[" "]
    state = _TeleopState(command=stop_cmd, updated_at=time.monotonic())
    stop_event = asyncio.Event()

    logging.info("Keyboard teleop started.")
    logging.info("Controls: W/S forward-back, A/D strafe, Q/E turn, R run, F sneak, Space stop, X/Esc exit.")

    async def _key_reader() -> None:
        while not stop_event.is_set():
            key = (await asyncio.to_thread(_read_key_blocking)).lower()
            if key in {"x", "\x1b"}:
                stop_event.set()
                return
            cmd = KEYBOARD_TELEOP_MAP.get(key)
            if cmd is None:
                continue
            state.command = cmd
            state.updated_at = time.monotonic()

    async def _control_loop() -> None:
        try:
            while not stop_event.is_set():
                if (time.monotonic() - state.updated_at) > KEYBOARD_TELEOP_IDLE_TIMEOUT_S:
                    cmd = stop_cmd
                else:
                    cmd = state.command
                await _run_locomotion(manip, cmd, source="sonic_keyboard")
        finally:
            await _run_locomotion(manip, stop_cmd, source="sonic_keyboard_stop")

    reader_task = asyncio.create_task(_key_reader())
    control_task = asyncio.create_task(_control_loop())
    done, pending = await asyncio.wait({reader_task, control_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    for task in done:
        try:
            await task
        except asyncio.CancelledError:
            pass
    logging.info("Keyboard teleop stopped.")


async def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)
    runtime_cfg = cfg.get("runtime", {})
    log_level = args.log_level or runtime_cfg.get("log_level", "INFO")
    _configure_logging(log_level)

    manip_cfg = dict(cfg.get("manipulation", {}))
    # Force locomotion-only mode (never route to GR00T action path).
    manip_cfg["backend"] = "mock"
    manip_cfg["mock_mode"] = True
    manip_cfg["fallback_to_mock"] = True

    manip = GrootManipulator(manip_cfg)
    try:
        await manip.warmup()
        if not bool(getattr(manip, "locomotion_enabled", False)):
            raise RuntimeError("No locomotion backend is enabled (direct_policy/SONIC).")
        adapter = getattr(manip, "_action_adapter", None)
        adapter_backend = getattr(adapter, "backend", "")
        if adapter_backend != "ros2_topic":
            raise RuntimeError(
                "G1 action adapter is not in ROS2 mode. Check rclpy environment and config "
                "(manipulation.action_adapter.backend must be 'ros2_topic')."
            )

        backend_active = getattr(manip, "locomotion_backend_active", "unknown")
        logging.info("Locomotion-only CLI started. backend=%s Type 'exit' to quit.", backend_active)
        logging.info("Examples: walk forward, run, turn left, go right, sneak, keyboard")

        async def _run_text_command(text: str) -> None:
            t = text.strip()
            if not t:
                return
            lowered = t.lower()
            if lowered in {"keyboard", "teleop", "kb"}:
                await _run_keyboard_teleop(manip)
                return
            cmd = _parse_locomotion_intent(t)
            if cmd is None:
                logging.warning("Unsupported command for SONIC-only CLI: %s", t)
                return
            ok = await _run_locomotion(manip, cmd, source="sonic_text")
            logging.info("locomotion_ok=%s text=%s cmd=%s", ok, t, cmd)

        if args.command:
            await _run_text_command(args.command)
        if args.keyboard:
            await _run_keyboard_teleop(manip)
        if args.no_interactive:
            return

        while True:
            try:
                user_input = await _async_input("sonic> ")
            except EOFError:
                break
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            await _run_text_command(user_input)
    finally:
        await manip.stop()
        logging.info("SONIC-only CLI stopped.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SONIC-only locomotion CLI (no GR00T actions).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to agent_runtime config.yaml",
    )
    parser.add_argument("--command", type=str, default="", help="Optional one-shot command before interactive mode")
    parser.add_argument("--keyboard", action="store_true", help="Start in keyboard teleop mode")
    parser.add_argument("--no-interactive", action="store_true", help="Run one-shot command and exit")
    parser.add_argument("--log-level", type=str, default="", help="Override log level (INFO, DEBUG, ...)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
