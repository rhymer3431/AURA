# Inference Subsystem

- Scope: managed inference stack for NavDP, InternVLA System2, planner LLM serving, health aggregation, and inference clients.
- Package root: `src/systems/inference`
- Modules:
  - `api`: `runtime.py`, `serve_inference_stack.py`
  - `client.py`
  - `navdp/*`
  - `system2/*`
  - `planner/*`
  - `stack/*`
- Entry points:
  - `python -m systems.inference.api.serve_inference_stack`
  - `python -m systems.inference.system2.check_session`
  - `scripts/run_system/inference_stack_windows.bat`
