# Navigation Subsystem

- Scope: runtime geometry, follower logic, goal providers, and the NavDP client surface consumed by control/world-state.
- Package root: `src/systems/navigation`
- Modules:
  - `api`: `runtime.py`, `geometry.py`, `navdp_sensors.py`
  - `geometry.py`
  - `goals.py`
  - `follower.py`
  - `client.py`
- Runtime note:
  - standalone NavDP server launching moved into the inference subsystem and is exposed through `systems.inference.navdp.server`
