from __future__ import annotations

import argparse

from apps.runtime_common import build_runtime_io
from runtime.memory_agent_runtime import MemoryAgentRuntime


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured memory agent with direct IPC wiring.")
    parser.add_argument("--memory-db-path", type=str, default="state/memory/memory.sqlite")
    parser.add_argument("--bootstrap-rule", type=str, default="find:apple:kitchen")
    parser.add_argument("--bus", type=str, choices=("inproc", "zmq"), default="inproc")
    parser.add_argument("--endpoint", type=str, default="tcp://127.0.0.1:5560")
    parser.add_argument("--bind", action="store_true")
    parser.add_argument("--agent-id", type=str, default="memory_agent")
    parser.add_argument("--control-endpoint", type=str, default="")
    parser.add_argument("--telemetry-endpoint", type=str, default="")
    parser.add_argument("--shm-name", type=str, default="isaac_aura_frames")
    parser.add_argument("--shm-slot-size", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--shm-capacity", type=int, default=8)
    parser.add_argument("--command", type=str, default="아까 봤던 사과를 찾아가")
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--loopback", action="store_true")
    parser.add_argument("--detector-engine-path", type=str, default="")
    parser.add_argument("--frame-source", type=str, choices=("auto", "live", "synthetic"), default="auto")
    parser.add_argument("--strict-live", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-interval-ms", type=int, default=100)
    parser.add_argument("--health-interval-sec", type=float, default=5.0)
    parser.add_argument("--persist-interval-sec", type=float, default=15.0)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--idle-exit-after-sec", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = None
    run_forever = False
    runtime_io = build_runtime_io(
        bus_kind=args.bus,
        endpoint=args.endpoint,
        bind=bool(args.bind),
        shm_name=args.shm_name,
        shm_slot_size=int(args.shm_slot_size),
        shm_capacity=int(args.shm_capacity),
        create_shm=bool(args.bind),
        role="bridge" if bool(args.bind) else "agent",
        control_endpoint=args.control_endpoint,
        telemetry_endpoint=args.telemetry_endpoint,
        identity="" if bool(args.bind) else str(args.agent_id),
    )
    try:
        runtime = MemoryAgentRuntime(args, bus=runtime_io.bus, shm_ring=runtime_io.shm_ring)
        run_forever = bool(args.serve) or (args.bus == "zmq" and not bool(args.loopback) and not bool(args.once))
        try:
            if run_forever:
                return runtime.run()
            result = runtime.run_once()
            snapshot = runtime.supervisor.snapshot()
            action_type = "none" if result.command is None else result.command.action_type
            print(
                "[MEMORY_AGENT] "
                f"bus={args.bus} state={snapshot['state']} detector={snapshot['detector_backend']} "
                f"action={action_type}"
            )
            return 0
        except RuntimeError as exc:
            print(f"[MEMORY_AGENT] {exc}")
            return 1
    finally:
        if runtime is not None:
            runtime.close(persist=run_forever)
        runtime_io.close(unlink_shm=bool(args.bind and args.bus == "zmq"))


if __name__ == "__main__":
    raise SystemExit(main())
