# Refactor Notes

## Current Layout
- Canonical code now lives directly under `src/` in function-oriented packages: `adapters`, `apps`, `common`, `control`, `inference`, `locomotion`, `runtime`, `services`, and `vendor`.
- The old wrapper package under `src/` has been removed.
- `navdp/`, `navdp_sidecar/`, and `g1_play/` remain only as compatibility surfaces.

## Ownership
- `src/runtime`: bridge orchestration, runtime sessions, CLI parsing, launcher-facing flow.
- `src/locomotion`: ONNX locomotion runtime, controller logic, G1 assets, play entrypoint.
- `src/common`: geometry and scene helpers shared across runtime and services.
- `src/adapters`: HTTP and sensor adapters.
- `src/control`: async planners and trajectory tracker.
- `src/apps`: Flask route modules and server main functions.
- `src/services`: request decoding and orchestration services.
- `src/inference`: policy agent, backbone, and policy network.
- `src/vendor`: bundled `depth_anything` code.

## Compatibility Policy
- Keep legacy imports and launchers working.
- Do not place real implementation back under provenance-based package roots.
- Put new code into the functional package that actually owns the responsibility.
