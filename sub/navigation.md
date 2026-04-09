# Navigation Subsystem

- Scope: NavDP client/server boundary, path planning output, goal geometry, follower state, and the navigation backend policy stack.
- Package root: `src/systems/navigation`
- Layers:
  - `api`: `navdp_server.py`, `runtime.py`, `geometry.py`, `navdp_sensors.py`
  - `domain`: `navdp_geometry.py`, `navdp_goals.py`, `navdp_follower.py`
  - `infrastructure`: `navdp_client.py`, `navdp_backend/*`
- Entry points:
  - `python -m systems.navigation.api.navdp_server`
  - `src/systems/navigation/bin/run_navdp_server_windows.bat`
