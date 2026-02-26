# Project Layout

This repository is organized by runtime responsibility so modules are easier to reuse and locate.

## Top-level conventions

- `apps/`: runtime applications and services
- `tests/`: executable tests and probes
- `tools/`: diagnostics and one-off inspection utilities
- `docs/`: architecture and operation documents

## Key paths

- `apps/agent_runtime/`: orchestration runtime and robot behavior modules
- `apps/isaacsim_runner/`: Isaac Sim runner
- `apps/services/planner_server/`: planner HTTP service
- `apps/sonic_policy_server/`: SONIC policy server + telemetry runtime
- `tests/pipeline/`: telemetry-driven pipeline tests
- `tests/probes/`: low-level SONIC communication probes
- `tests/unit/`: fast unit tests
- `tools/diagnostics/`: mapping/model/joint inspection tools
- `docs/reports/`: generated scan artifacts kept for reference

## Compatibility entry points

- `sonic_policy_server.py`: compatibility launcher that forwards to `apps/sonic_policy_server/server.py`
- `telemetry_runtime.py`: compatibility import shim forwarding to `apps/sonic_policy_server/telemetry_runtime.py`
