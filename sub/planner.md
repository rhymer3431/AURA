# Planner Subsystem

- Scope: task-frame generation, planner endpoint calls, ontology/normalization, schema validation, subgoal orchestration, and reporting.
- Package root: `src/systems/planner`
- Layers:
  - `api`: `runtime.py`
  - `application`: `aura_adapter.py`, `orchestration.py`, `planner_service.py`
  - `domain`: `schemas.py`, `task_frames.py`, `normalizer.py`, `ontology.py`, `validator.py`, `reporting.py`
  - `infrastructure`: `llm_client.py`
- Entry points:
  - `scripts/serve_planner_qwen3_nothink.ps1`
