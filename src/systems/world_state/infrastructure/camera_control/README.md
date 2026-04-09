# Camera Pitch Control

This directory contains the code that controls the G1 camera pitch at runtime.

## Files

- `api.py`
  - Local HTTP API for `GET/POST /camera/pitch`
- `sensor.py`
  - Camera attachment, RGB-D capture, and pitch application logic
- `runtime_service.py`
  - Shared service used by standalone runtime modes
- `targeting.py`
  - Chooses which camera rig or prim is controlled

## Control Target Selection

The pitch API selects the control target in this order:

1. `--camera_prim_path` if explicitly provided
2. `<robot_prim_path>/head_link/Realsense` when the existing D455 rig is present
3. `<robot_prim_path>/NavCamera` as a runtime fallback camera

If the chosen path is a camera rig root instead of a `Camera` prim, `sensor.py` keeps pitch control on the rig root and discovers a descendant `Camera` prim for RGB-D streaming.

## API

- `GET /healthz`
- `GET /camera/pitch`
- `POST /camera/pitch`

Examples:

```bash
curl http://127.0.0.1:8891/camera/pitch
curl -X POST http://127.0.0.1:8891/camera/pitch -H "Content-Type: application/json" -d '{"pitch_deg":-15}'
curl -X POST http://127.0.0.1:8891/camera/pitch -H "Content-Type: application/json" -d '{"delta_deg":5}'
```

Pitch convention:

- positive value: look upward
- negative value: look downward

## Runtime Integration

- `systems.control.application.runtime_orchestrator`
  - Uses `RuntimeCameraPitchService` while the main control runtime is active
- `systems.navigation.api.navdp_sensors`
  - Uses the same package while NavDP camera capture is active

There are no compatibility wrappers for the removed `g1_play` package. Active runtime imports now resolve through `systems.world_state.api.*`.
