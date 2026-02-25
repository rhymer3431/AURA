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
from typing import Any, Dict, Optional, Tuple

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
        format="%(asctime)s [%(levelname)s] [groot_cli] %(message)s",
    )


async def _async_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def _parse_command(line: str) -> Tuple[str, str, str]:
    text = line.strip()
    if not text:
        return "noop", "", ""

    lowered = text.lower()
    if lowered.startswith("pick "):
        target = text[5:].strip()
        return "pick", target, ""
    if lowered.startswith("inspect "):
        target = text[8:].strip()
        return "inspect", target, ""
    if lowered in {"keyboard", "teleop", "kb"}:
        return "keyboard", "", ""
    if _parse_locomotion_intent(text) is not None:
        return "locomotion", "", text
    return "instruction", "", text


def _parse_locomotion_intent(text: str) -> Optional[SonicVelocityCommand]:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    for pattern, cmd in LOCOMOTION_MAP.items():
        if pattern in compact:
            return cmd
    return None


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
                manip.set_sonic_velocity(cmd)
                await manip.execute_locomotion(
                    linear_x=float(cmd.vx),
                    linear_y=float(cmd.vy),
                    angular_z=float(cmd.yaw_rate),
                    duration_s=float(cmd.duration_s),
                    source="keyboard_teleop",
                )
        finally:
            manip.set_sonic_velocity(stop_cmd)
            await manip.execute_locomotion(
                linear_x=0.0,
                linear_y=0.0,
                angular_z=0.0,
                duration_s=float(stop_cmd.duration_s),
                source="keyboard_teleop_stop",
            )

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
    manip = GrootManipulator(manip_cfg)
    warmup_ok = True
    try:
        await manip.warmup()
    except Exception as exc:
        warmup_ok = False
        logging.warning(
            "GR00T warmup failed: %s. Locomotion commands are still available.",
            exc,
        )
    logging.info("Realtime GR00T CLI started. Type 'exit' to quit.")
    logging.info(
        "Command examples: 'pick apple', 'inspect apple', 'walk forward', 'run', 'go left', 'sneak', 'keyboard'."
    )

    async def _run_text_command(text: str) -> None:
        mode, target, instruction = _parse_command(text)
        if mode == "noop":
            return
        if mode == "pick":
            if not warmup_ok:
                logging.warning("GR00T policy backend is not ready. 'pick' is unavailable.")
                return
            if not target:
                logging.warning("pick command requires target. Example: pick apple")
                return
            ok = await manip.pick(target=target, instruction=f"Pick up the {target} and place it on the plate.")
            logging.info("pick_ok=%s target=%s", ok, target)
            return
        if mode == "inspect":
            if not warmup_ok:
                logging.warning("GR00T policy backend is not ready. 'inspect' is unavailable.")
                return
            if not target:
                logging.warning("inspect command requires target. Example: inspect apple")
                return
            ok = await manip.inspect(target=target, instruction=f"Inspect the {target}.")
            logging.info("inspect_ok=%s target=%s", ok, target)
            return
        if mode == "keyboard":
            await _run_keyboard_teleop(manip)
            return
        if mode == "locomotion":
            sonic_cmd = _parse_locomotion_intent(instruction)
            if sonic_cmd is None:
                logging.warning("Unsupported locomotion command: %s", instruction)
                return
            manip.set_sonic_velocity(sonic_cmd)
            ok = await manip.execute_locomotion(
                linear_x=float(sonic_cmd.vx),
                linear_y=float(sonic_cmd.vy),
                angular_z=float(sonic_cmd.yaw_rate),
                duration_s=float(sonic_cmd.duration_s),
                source="cli_locomotion",
            )
            logging.info("locomotion_ok=%s text=%s sonic_cmd=%s", ok, instruction, sonic_cmd)
            return

        if not warmup_ok:
            logging.warning("GR00T policy backend is not ready. Instruction execution is unavailable.")
            return
        ok = await manip.execute_instruction(instruction=instruction)
        logging.info("instruction_ok=%s text=%s", ok, instruction)

    try:
        if args.command:
            await _run_text_command(args.command)
        if args.keyboard:
            await _run_keyboard_teleop(manip)

        if args.no_interactive:
            return

        while True:
            try:
                user_input = await _async_input("groot> ")
            except EOFError:
                break
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            await _run_text_command(user_input)
    finally:
        await manip.stop()
        logging.info("Realtime GR00T CLI stopped.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime text command CLI for GR00T manipulator.")
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
