# IPC Protocol

## Default Principle
- Default runtime path is direct IPC, not Flask/HTTP.
- Small control messages use `MessageBus`.
- Large RGB/depth payloads prefer `SharedMemoryRing`.
- In-process debug mode can fall back to inline frame payloads when shared memory is not configured.

## Transports
- `ipc.inproc_bus.InprocBus`
  - Default for local stack and loopback smoke tests.
  - Deterministic and test-friendly.
- `ipc.zmq_bus.ZmqBus`
  - Current two-process localhost transport.
  - Uses a `PAIR` socket for the current 2-process bridge/agent topology.
  - Requires `pyzmq`.
- `ipc.shm_ring.SharedMemoryRing`
  - Stores encoded RGB/depth arrays and passes `ShmSlotRef` handles through `FrameHeader.metadata`.

## Message Types
- `FrameHeader`
  - `frame_id`, `timestamp_ns`, `source`
  - `camera_pose_xyz`, `camera_quat_wxyz`
  - shared-memory or inline frame references in `metadata`
- `TaskRequest`
  - natural-language command plus `target_json` and optional `speaker_id`
- `ActionCommand`
  - Common command set:
    - `STOP`
    - `LOOK_AT`
    - `FOLLOW_TARGET`
    - `NAV_TO_PLACE`
    - `NAV_TO_POSE`
    - `LOCAL_SEARCH`
- `ActionStatus`
  - execution feedback from bridge/executor back to the orchestrator
  - includes `state`, `reason`, `robot_pose_xyz`, and `distance_remaining_m`

## Topics
- `isaac.task`
  - task text from local stack, CLI, or bridge process
- `isaac.observation`
  - `FrameHeader` from Isaac bridge to memory agent
- `isaac.command`
  - `ActionCommand` from orchestrator to executor
- `isaac.status`
  - `ActionStatus` from executor back to orchestrator

## Actual Flow
- Single-process loopback
  - `apps.local_stack_app` or `apps.memory_agent_app --loopback`
  - `TaskRequest -> Perception -> Memory -> ActionCommand`
- Two-process local stack
  1. `apps.memory_agent_app --bus zmq --bind`
  2. `apps.isaac_bridge_app --bus zmq --connect`
  3. Bridge publishes `TaskRequest` and `FrameHeader`
  4. Memory agent polls bus, reconstructs frames, updates memory, and publishes `ActionCommand`
  5. Bridge drains `ActionCommand`

## Current Limits
- `PAIR` is enough for the current 2-process topology, but pub/sub or router/dealer is a likely next step for richer fan-out.
- In shared-memory mode, both processes must agree on shm name, slot size, and capacity.
