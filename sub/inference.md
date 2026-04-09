# Inference Subsystem

- Scope: InternVLA grounding, llama.cpp-backed HTTP serving, sidecar/session health management, and multimodal inference helpers.
- Package root: `src/systems/inference`
- Layers:
  - `api`: `serve_internvla_nav_server.py`, `check_internvla_session.py`, `runtime.py`
  - `infrastructure`: `internvla_nav.py`
- Entry points:
  - `python -m systems.inference.api.serve_internvla_nav_server`
  - `python -m systems.inference.api.check_internvla_session`
  - `src/systems/inference/bin/run_internvla_nav_server_windows.bat`
  - `src/systems/inference/bin/check_internvla_session_windows.bat`
