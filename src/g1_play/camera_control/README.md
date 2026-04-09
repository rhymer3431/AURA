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

- `g1_play/runtime.py`
  - Uses `RuntimeCameraPitchService` for non-NavDP modes
- `g1_play/navdp_runtime.py`
  - Uses the same package directly while NavDP is active

## Backward Compatibility

The older modules below are left as thin wrappers so existing imports do not break immediately:

- `g1_play/camera_api.py`
- `g1_play/camera_runtime.py`
- `g1_play/navdp_sensors.py`
