# IPC Protocol

## Default Principle
- Default runtime path is direct IPC, not Flask/HTTP.
- Small control/state messages use `MessageBus`.
- Large RGB/depth payloads prefer `SharedMemoryRing`.

## Transports
- `InprocBus`
  - single-process debug and loopback
- `ZmqBus`
  - control plane: bridge `ROUTER`, agent `DEALER`
  - telemetry plane: bridge `PUB`, agent `SUB`
  - retained control replay for late-joining agents
- `SharedMemoryRing`
  - RGB/depth payload slots referenced from `FrameHeader.metadata`

## Core Message Types
- `TaskRequest`
- `FrameHeader`
- `ActionCommand`
- `ActionStatus`
- `CapabilityReport`
- `RuntimeNotice`
- `HealthPing`

## Command Set
- `STOP`
- `LOOK_AT`
- `FOLLOW_TARGET`
- `NAV_TO_PLACE`
- `NAV_TO_POSE`
- `LOCAL_SEARCH`

## Standard Runtime Flow
1. Bridge publishes `FrameHeader` and shared-memory references.
2. Memory agent reconstructs the batch and runs perception -> memory -> orchestration.
3. Memory agent publishes `ActionCommand`.
4. Bridge executes low-level subgoals and publishes `ActionStatus`.

## Live Smoke Parity
- `apps.live_smoke_app` is not the normal bridge runtime, but it intentionally reuses the same batch shape.
- That keeps live diagnostics aligned with the production `FrameHeader`/observation path.

## Live Smoke Artifacts
The diagnostics JSON records:
- selected launch mode
- deprecated alias use, if any
- selected bootstrap profile
- smoke target tier
- current and failed phase
- compatibility report
- structured recommendation items
- smoke tier result summary

Additional artifacts:
- CLI args
- enabled extensions
- D455 asset resolution
- D455 mount report
- stage prim tree
- sensor init report
- first frame report
- smoke metrics

## Smoke Tier Semantics
- `sensor_status`
  - proves D455/frame ingress
- `pipeline_status`
  - proves perception ingress
  - empty detection batch is allowed
- `memory_status`
  - proves memory update path

These are stored separately so a blank scene does not look like total smoke failure.

## Compatibility and Recommendation Layer
Preflight and smoke generate a structured compatibility report:
- environment availability
- experience and assets availability
- extension support
- likely runtime mismatch
- recommended launch mode/profile

Recommendation items are derived from:
- compatibility report
- last failed phase
- smoke tier result

This is what drives “retry standalone with warmup”, “switch to editor_assisted”, or “treat as empty-scene pipeline pass”.

## Current Limits
- Multi-agent fan-out is retained replay plus shared-topic merge, not targeted routing.
- `editor_assisted` and `extension_mode` require code to run inside an existing Full App / Kit process.
