# Perception Subsystem

- Scope: camera APIs, camera pitch runtime service, camera prim attachment, and camera sensor capture.
- Package root: `src/systems/perception`
- Layers:
  - `api`: `camera_api.py`
  - `application`: `camera_runtime.py`
  - `infrastructure`: `camera_control/*`
- Notes:
  - The runtime now controls a child camera prim instead of rotating the articulation-linked rig root while physics is running.
  - This avoids the PhysX `copyInternalStateToCache()` warning triggered by in-sim articulation cache writes.
