from __future__ import annotations

from dataclasses import dataclass

from apps.frame_bridge_app import parse_args
from apps.runtime_common import RuntimeIo, build_runtime_io
from runtime.frame_editor_bridge import AttachedFrameBridgeRuntime, editor_bridge_available


@dataclass
class AttachedFrameBridgeSession:
    runtime: AttachedFrameBridgeRuntime
    runtime_io: RuntimeIo
    unlink_shm: bool = False

    def tick(self, frame_idx: int | None = None):
        return self.runtime.tick(frame_idx)

    def close(self) -> None:
        self.runtime.close()
        self.runtime_io.close(unlink_shm=self.unlink_shm)


def attach_current_stage(
    *,
    controller,
    argv: list[str] | None = None,
    simulation_app=None,
    stage=None,
) -> AttachedFrameBridgeSession:
    available, reason = editor_bridge_available()
    if not available:
        raise RuntimeError(reason)
    args = parse_args(argv or [])
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
    runtime = AttachedFrameBridgeRuntime(args=args, controller=controller, bus=runtime_io.bus, shm_ring=runtime_io.shm_ring)
    try:
        runtime.start(simulation_app=simulation_app, stage=stage)
    except Exception:
        runtime_io.close(unlink_shm=bool((not args.connect) and args.bus == "zmq"))
        raise
    return AttachedFrameBridgeSession(
        runtime=runtime,
        runtime_io=runtime_io,
        unlink_shm=bool((not args.connect) and args.bus == "zmq"),
    )
