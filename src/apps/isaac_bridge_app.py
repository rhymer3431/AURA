from __future__ import annotations

import argparse
import importlib
import time

from apps.runtime_common import build_frame_source, build_runtime_io, frame_sample_to_batch, infer_demo_scene
from ipc.messages import HealthPing, RuntimeNotice, TaskRequest
from locomotion.args import BOOTSTRAP_PARSER, add_runtime_args
from runtime.g1_bridge_args import add_subgoal_executor_args
from runtime.isaac_bridge_runtime import run_live_isaac_bridge
from runtime.supervisor import Supervisor, SupervisorConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Isaac bridge launcher with IPC wiring.", parents=[BOOTSTRAP_PARSER])
    add_runtime_args(parser)
    parser.set_defaults(env_url="/Isaac/Environments/Simple_Warehouse/warehouse.usd")
    add_subgoal_executor_args(parser)
    parser.add_argument("--command", type=str, default="아까 봤던 사과를 찾아가")
    parser.add_argument("--bus", type=str, choices=("inproc", "zmq"), default="inproc")
    parser.add_argument("--endpoint", type=str, default="tcp://127.0.0.1:5560")
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--control-endpoint", type=str, default="")
    parser.add_argument("--telemetry-endpoint", type=str, default="")
    parser.add_argument("--shm-name", type=str, default="isaac_aura_frames")
    parser.add_argument("--shm-slot-size", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--shm-capacity", type=int, default=8)
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--loopback", action="store_true")
    parser.add_argument("--frame-source", type=str, choices=("auto", "live", "synthetic"), default="auto")
    parser.add_argument("--strict-live", action="store_true")
    parser.add_argument("--sensor-report-path", type=str, default="")
    return parser.parse_args(argv)


def isaac_bootstrap_available() -> tuple[bool, str]:
    try:
        importlib.import_module("isaacsim")
    except Exception as exc:  # noqa: BLE001
        return False, f"Isaac standalone bootstrap unavailable: {type(exc).__name__}: {exc}"
    return True, ""


def run_lightweight_bridge(
    args: argparse.Namespace,
    *,
    force_synthetic: bool = False,
    initial_notice: str = "",
) -> int:
    frame_source = None
    runtime_io = build_runtime_io(
        bus_kind=args.bus,
        endpoint=args.endpoint,
        bind=not bool(args.connect),
        shm_name=args.shm_name,
        shm_slot_size=int(args.shm_slot_size),
        shm_capacity=int(args.shm_capacity),
        create_shm=not bool(args.connect),
        role="bridge",
        control_endpoint=args.control_endpoint,
        telemetry_endpoint=args.telemetry_endpoint,
    )
    try:
        from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter, IsaacBridgeAdapterConfig

        bridge = IsaacBridgeAdapter(runtime_io.bus, IsaacBridgeAdapterConfig(), shm_ring=runtime_io.shm_ring)
        bridge.publish_health(HealthPing(component="isaac_bridge", details={"bus": args.bus, "mode": "lightweight"}))
        if initial_notice != "":
            bridge.publish_notice(
                RuntimeNotice(
                    component="isaac_bridge",
                    level="warning",
                    notice=initial_notice,
                    details={"frame_source_mode": "synthetic" if force_synthetic else args.frame_source},
                )
            )
        scene = str(args.scene).strip() or infer_demo_scene(args.command)
        frame_source = build_frame_source(
            mode="synthetic" if force_synthetic else args.frame_source,
            scene=scene,
            source_name="isaac_bridge",
            room_id="kitchen" if scene == "apple" else "",
            strict_live=bool(args.strict_live),
        )
        source_report = frame_source.start()
        if args.frame_source == "live" and source_report.status != "ready":
            print(f"[ISAAC_BRIDGE] live frame source unavailable: {source_report.notice}")
            return 1
        bridge.publish_notice(
            RuntimeNotice(
                component="isaac_bridge",
                level="warning" if source_report.fallback_used else "info",
                notice=source_report.notice or f"frame source status={source_report.status}",
                details={
                    "frame_source": source_report.source_name,
                    "fallback_used": source_report.fallback_used,
                    **source_report.details,
                },
            )
        )
        if bool(args.loopback) or args.bus == "inproc":
            supervisor = Supervisor(
                bus=runtime_io.bus,
                shm_ring=runtime_io.shm_ring,
                config=SupervisorConfig(
                    memory_db_path=args.memory_db_path,
                    detector_engine_path=args.detector_engine_path,
                ),
            )
            bridge.publish_task_request(TaskRequest(command_text=str(args.command)))
            sample = frame_source.read()
            if sample is not None:
                bridge.publish_observation_batch(frame_sample_to_batch(sample))
            supervisor.run_bus_cycle(now=time.time())
        else:
            bridge.publish_task_request(TaskRequest(command_text=str(args.command)))
            sample = frame_source.read()
            if sample is not None:
                bridge.publish_observation_batch(frame_sample_to_batch(sample))

        commands = bridge.drain_commands()
        latest = commands[-1] if commands else None
        print(
            "[ISAAC_BRIDGE] "
            f"bus={args.bus} mode=lightweight command={latest.action_type if latest else 'none'} "
            f"shm={'on' if runtime_io.shm_ring is not None else 'off'}"
        )
        return 0
    finally:
        if frame_source is not None:
            try:
                frame_source.close()
            except Exception:  # noqa: BLE001
                pass
        runtime_io.close(unlink_shm=bool((not args.connect) and args.bus == "zmq"))


def run_live_bridge(args: argparse.Namespace) -> int:
    runtime_io = build_runtime_io(
        bus_kind=args.bus,
        endpoint=args.endpoint,
        bind=not bool(args.connect),
        shm_name=args.shm_name,
        shm_slot_size=int(args.shm_slot_size),
        shm_capacity=int(args.shm_capacity),
        create_shm=not bool(args.connect),
        role="bridge",
        control_endpoint=args.control_endpoint,
        telemetry_endpoint=args.telemetry_endpoint,
    )
    try:
        return run_live_isaac_bridge(args, bus=runtime_io.bus, shm_ring=runtime_io.shm_ring)
    finally:
        runtime_io.close(unlink_shm=bool((not args.connect) and args.bus == "zmq"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if str(args.frame_source).lower() == "synthetic":
        return run_lightweight_bridge(args)

    live_available, live_reason = isaac_bootstrap_available()
    if live_available:
        return run_live_bridge(args)
    if str(args.frame_source).lower() == "live":
        print(f"[ISAAC_BRIDGE] live bridge unavailable: {live_reason}")
        print("[ISAAC_BRIDGE] use scripts/powershell/run_live_smoke_preflight.ps1 or run_live_smoke.ps1 for phase diagnostics.")
        return 1
    if live_reason != "":
        print(f"[ISAAC_BRIDGE] live bootstrap unavailable, falling back to synthetic path: {live_reason}")
        print("[ISAAC_BRIDGE] use scripts/powershell/run_live_smoke_preflight.ps1 to diagnose the live bootstrap path.")
    return run_lightweight_bridge(args, force_synthetic=True, initial_notice=live_reason)


if __name__ == "__main__":
    raise SystemExit(main())
