# Function-Oriented `src/` Refactor

## Summary
- Before: canonical code lived under `src/navdp_app/`, while `navdp/`, `navdp_sidecar/`, and `g1_play/` acted as partially overlapping compatibility roots.
- After: canonical code lives directly under `src/` in function-oriented packages only.
- Compatibility remains only where public imports or launchers still need it.

## Before Vs After

### Before
```text
src/
└── navdp_app/
    ├── adapters/
    ├── apps/
    ├── common/
    ├── control/
    ├── inference/
    ├── locomotion/
    ├── runtime/
    ├── services/
    └── vendor/
navdp/
navdp_sidecar/
g1_play/
```

### After
```text
src/
├── adapters/
├── apps/
├── common/
├── control/
├── inference/
├── locomotion/
├── runtime/
├── services/
└── vendor/
navdp/            # compatibility only
navdp_sidecar/    # compatibility only
g1_play/          # compatibility only
```

## Exact Module Migration Mapping

| Before | After |
| --- | --- |
| `src/navdp_app/adapters/*` | `src/adapters/*` |
| `src/navdp_app/apps/*` | `src/apps/*` |
| `src/navdp_app/common/*` | `src/common/*` |
| `src/navdp_app/control/*` | `src/control/*` |
| `src/navdp_app/inference/*` | `src/inference/*` |
| `src/navdp_app/locomotion/*` | `src/locomotion/*` |
| `src/navdp_app/runtime/*` | `src/runtime/*` |
| `src/navdp_app/services/*` | `src/services/*` |
| `src/navdp_app/vendor/*` | `src/vendor/*` |
| `g1_play/g1/config.py` | `src/locomotion/g1/config.py` |
| `navdp_sidecar/depth_anything/*` | removed in favor of `src/vendor/depth_anything/*` |

## Top-Level Package Rationale
- `src/adapters`: outbound HTTP clients, sensor capture, and boundary-specific payload handling.
- `src/apps`: executable Flask modules and route registration.
- `src/common`: reusable geometry and scene utilities shared across runtime, control, and services.
- `src/control`: planners and tracking logic used by runtime coordination.
- `src/inference`: model-specific policy code only.
- `src/locomotion`: ONNX locomotion runtime, controller code, and G1 assets.
- `src/runtime`: orchestration and launcher-facing flow.
- `src/services`: service-layer orchestration and request handling.
- `src/vendor`: isolated bundled third-party code.

## Entrypoint Mapping

| Surface | Canonical Target |
| --- | --- |
| `scripts/powershell/run_aura_runtime.ps1` | `python -m runtime.aura_runtime` |
| `scripts/powershell/run_g1_pointgoal.ps1` | compatibility wrapper to `scripts/powershell/run_aura_runtime.ps1` |
| `scripts/powershell/run_navdp_server.ps1` | `python -m apps.navdp_server_app` |
| `scripts/powershell/run_vlm_dual_server.ps1` | `python -m apps.dual_server_app` |
| `play_g1_keyboard_onnx.py` | `locomotion.entrypoint.main` |
| `navdp_sidecar/navdp_server.py` | compatibility wrapper to `apps.navdp_server_app` |
| `navdp_sidecar/vlm_dual_server.py` | compatibility wrapper to `apps.dual_server_app` |

## Remaining Compatibility Layers
- `navdp/*`: legacy import wrappers for bridge/runtime/common/control/adapter surfaces.
- `navdp_sidecar/*`: legacy import and script wrappers for sidecar apps and services.
- `g1_play/*`: legacy import wrappers for locomotion entrypoints and assets.
- `g1_play/g1/g1_d455.usd`: compatibility asset copy retained as a fallback path for existing launchers.

## Removed Vs Preserved Legacy Surfaces
- Removed: canonical `src/navdp_app/` package tree.
- Removed: duplicated `navdp_sidecar/depth_anything/` vendor tree.
- Preserved: root PowerShell compatibility launchers.
- Preserved: `navdp_sidecar/navdp_server.py`, `navdp_sidecar/vlm_dual_server.py`, and `play_g1_keyboard_onnx.py`.

## Follow-Up Cleanup Opportunities
- Replace remaining wildcard compatibility re-exports with explicit symbol exports if import contracts need tighter control.
- Add targeted tests for the compatibility wrappers themselves.
- If the repo becomes a full git workspace again, add package-install and launcher smoke checks to CI.
