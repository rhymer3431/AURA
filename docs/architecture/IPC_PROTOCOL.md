# IPC Protocol

## Design Goals
- Default control path is direct IPC, not Flask/HTTP orchestration.
- Large frame payloads are intended for shared memory ring buffers.
- Small control/status messages use a bus abstraction that supports in-process debug and future localhost sockets.

## Implemented Transports
- `ipc.inproc_bus.InprocBus`
  - Default single-process debug transport.
  - Deterministic for tests and local stack scaffolding.
- `ipc.zmq_bus.ZmqBus`
  - Optional localhost transport with a `PAIR` socket skeleton.
  - Requires `pyzmq`.
- `ipc.shm_ring.SharedMemoryRing`
  - Shared-memory slot ring for frame bytes.
  - Returns `ShmSlotRef` handles instead of copying payloads through the message bus.

## Message Types
- `FrameHeader`
  - `frame_id`, `timestamp_ns`, `source`
  - Shared-memory references for RGB/depth
  - Camera pose metadata
- `ActionCommand`
  - `LOOK_AT`, `FOLLOW_PERSON`, `NAV_TO_PLACE`, `NAV_TO_POSE`, `LOCAL_SEARCH`, `STOP`
  - Carries task id plus target object/place/pose info
- `ActionStatus`
  - Execution state from the bridge back to the orchestrator
  - Includes robot pose and distance remaining when available
- `TaskRequest`
  - Natural-language task request with optional target JSON and speaker binding

## Topics
- `isaac.observation`
  - Bridge publishes `FrameHeader`
- `isaac.task`
  - Bridge or local stack publishes `TaskRequest`
- `isaac.command`
  - Orchestrator publishes `ActionCommand`
- `isaac.status`
  - Bridge publishes `ActionStatus`

## Encoding
- `ipc.codec.encode_message()` serializes messages as compact JSON bytes.
- `ipc.codec.decode_message()` restores typed dataclasses.
- Tests cover codec round-trips for all mandatory message types.

## Planned Next Step
- Replace `PAIR` with pub/sub or router/dealer once cross-process Isaac bridge and memory agent are wired together on Windows.
