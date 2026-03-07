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
  - Uses a 2-plane topology:
    - control plane: bridge `ROUTER` bind, memory agent `DEALER` connect
    - telemetry plane: bridge `PUB` bind, memory agent `SUB` connect
  - Requires `pyzmq`.
  - Buffers control messages on the bridge until the agent has registered over the control plane.
- `ipc.shm_ring.SharedMemoryRing`
  - Stores encoded RGB/depth arrays and passes `ShmSlotRef` handles through `FrameHeader.metadata`.
- `ipc.transport_health.TransportHealthTracker`
  - Tracks last send/receive timestamps, reconnect attempts, queued messages, dropped messages, and last error per plane.

## Message Types
- `FrameHeader`
  - `frame_id`, `timestamp_ns`, `source`
  - `camera_pose_xyz`, `camera_quat_wxyz`
  - `robot_pose_xyz`, `robot_yaw_rad`, `sim_time_s`
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
  - Carries `target_person_id` and semantic hint metadata when available
- `ActionStatus`
  - execution feedback from bridge/executor back to the orchestrator
  - includes `state`, `reason`, `robot_pose_xyz`, and `distance_remaining_m`
- `CapabilityReport`
  - structured runtime diagnostics such as detector capability reports
  - live bridge also emits structured D455/camera initialization diagnostics
- `RuntimeNotice`
  - human-readable runtime notices, including live-frame fallback notices
- `HealthPing`
  - lightweight process and transport liveness message

## Topics
- `isaac.task`
  - control plane
  - task text from local stack, CLI, or bridge process
- `isaac.observation`
  - telemetry plane
  - `FrameHeader` from Isaac bridge to memory agent
- `isaac.command`
  - control plane
  - `ActionCommand` from orchestrator to executor
- `isaac.status`
  - telemetry plane
  - `ActionStatus` from executor back to orchestrator
- `isaac.notice`
  - control plane
  - `RuntimeNotice`
- `isaac.capability`
  - control plane
  - `CapabilityReport`
- `isaac.health`
  - telemetry plane from bridge to agent
  - control-plane-compatible when emitted from the agent for registration/health

## Actual Flow
- Single-process loopback
  - `apps.local_stack_app` or `apps.memory_agent_app --loopback`
  - `TaskRequest -> Perception -> Memory -> ActionCommand`
- Two-process local stack
  1. Start bridge with bound control/telemetry endpoints
  2. In live mode, the bridge process owns standalone `SimulationApp` and initializes the D455/live RGB-D path before entering the locomotion loop
  3. Start memory agent and connect to the bridge
  4. Memory agent publishes `CapabilityReport`/`HealthPing` to register on the control plane
  5. Bridge publishes `TaskRequest` on control and `FrameHeader` on telemetry
  6. RGB/depth payloads travel inline or through `SharedMemoryRing` references carried in `FrameHeader.metadata`
  7. Memory agent reconstructs frames, updates memory, and publishes `ActionCommand` on control
  8. Bridge drains `ActionCommand`, executes `NAV_TO_POSE` / `NAV_TO_PLACE` / `LOCAL_SEARCH` / `LOOK_AT` / `STOP` locally, and publishes `ActionStatus` on telemetry
  9. `RuntimeNotice` and `CapabilityReport` remain on the control plane so late-connecting agents can still observe startup diagnostics

## Current Limits
- Current topology is single bridge plus single memory agent. Multi-agent fan-out is not wired yet.
- In shared-memory mode, both processes must agree on shm name, slot size, and capacity.
- Telemetry from the memory agent still reuses the control plane when needed; the primary telemetry direction remains bridge -> agent.
