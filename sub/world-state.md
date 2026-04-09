# World State Subsystem

- Scope: camera sensing, scene/asset lookup, camera pitch control, observation layout, and world-facing runtime state capture.
- Package root: `src/systems/world_state`
- Layers:
  - `api`: `camera_api.py`, `paths.py`, `scene.py`, `observation_layout.py`
  - `application`: `camera_runtime.py`
  - `domain`: `constants.py`, `paths.py`, `observation_layout.py`
  - `infrastructure`: `camera_control/*`, `scene.py`
- Assets:
  - `robots/g1/g1_d455.usd`
  - `tuned/params/env.yaml`
