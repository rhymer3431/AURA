from __future__ import annotations

import argparse
import time

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter, IsaacBridgeAdapterConfig
from apps.runtime_common import build_demo_batch, build_runtime_io, infer_demo_scene
from ipc.messages import TaskRequest
from runtime.supervisor import Supervisor, SupervisorConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured memory agent with direct IPC wiring.")
    parser.add_argument("--memory-db-path", type=str, default="state/memory/memory.sqlite")
    parser.add_argument("--bootstrap-rule", type=str, default="find:apple:kitchen")
    parser.add_argument("--bus", type=str, choices=("inproc", "zmq"), default="inproc")
    parser.add_argument("--endpoint", type=str, default="tcp://127.0.0.1:5560")
    parser.add_argument("--bind", action="store_true")
    parser.add_argument("--shm-name", type=str, default="isaac_aura_frames")
    parser.add_argument("--shm-slot-size", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--shm-capacity", type=int, default=8)
    parser.add_argument("--command", type=str, default="아까 봤던 사과를 찾아가")
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--loopback", action="store_true")
    parser.add_argument("--detector-engine-path", type=str, default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_io = build_runtime_io(
        bus_kind=args.bus,
        endpoint=args.endpoint,
        bind=bool(args.bind),
        shm_name=args.shm_name,
        shm_slot_size=int(args.shm_slot_size),
        shm_capacity=int(args.shm_capacity),
        create_shm=bool(args.bind),
    )
    try:
        supervisor = Supervisor(
            bus=runtime_io.bus,
            shm_ring=runtime_io.shm_ring,
            config=SupervisorConfig(
                memory_db_path=args.memory_db_path,
                detector_engine_path=args.detector_engine_path,
            ),
        )
        supervisor.memory_service.semantic_store.remember_rule(
            args.bootstrap_rule,
            "prioritize table-like surfaces",
            succeeded=True,
            metadata={"source": "bootstrap"},
        )

        bridge = IsaacBridgeAdapter(runtime_io.bus, IsaacBridgeAdapterConfig(), shm_ring=runtime_io.shm_ring)
        if bool(args.loopback) or args.bus == "inproc":
            scene = str(args.scene).strip() or infer_demo_scene(args.command)
            bridge.publish_task_request(TaskRequest(command_text=str(args.command)))
            bridge.publish_observation_batch(
                build_demo_batch(
                    frame_id=1,
                    scene=scene,
                    source="memory_agent_loopback",
                    room_id="kitchen" if scene == "apple" else "",
                )
            )
            command = supervisor.run_bus_cycle(now=time.time())
            commands = bridge.drain_commands()
            emitted = commands[-1] if commands else command
            snapshot = supervisor.snapshot()
            print(
                "[MEMORY_AGENT] "
                f"bus={args.bus} state={snapshot['state']} detector={snapshot['detector_backend']} "
                f"action={emitted.action_type if emitted else 'none'}"
            )
            return 0

        command = supervisor.run_bus_cycle(now=time.time())
        snapshot = supervisor.snapshot()
        print(
            "[MEMORY_AGENT] "
            f"bus={args.bus} state={snapshot['state']} detector={snapshot['detector_backend']} "
            f"action={command.action_type if command else 'none'}"
        )
        return 0
    finally:
        runtime_io.close(unlink_shm=bool(args.bind and args.bus == 'zmq'))


if __name__ == "__main__":
    raise SystemExit(main())
