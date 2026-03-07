# Memory Architecture

## Default Path
- Default runtime path is direct IPC plus external structured memory.
- Low-level locomotion stays in `runtime.g1_bridge` / `planning_session`.
- Task semantics stay in `services.task_orchestrator`.

## Stores
- `SpatialMemoryStore`
  - anchored `PlaceNode` / `ObjectNode`
  - object recall resolves object -> place -> nav target
- `TemporalMemoryStore`
  - speaker events, speaker bindings, follow loss, re-id candidates, recovery actions
- `EpisodicMemoryStore`
  - structured task records with candidate objects/places and recovery trace
- `SemanticMemoryStore`
  - rule-like memories with support count and success rate
- `WorkingMemory`
  - active subset only
  - uses recency/reachability/confidence plus semantic bonuses

## Online Update Flow
1. Detector/tracker/depth projection produce `ObsObject`
2. `MemoryService.observe_objects()` updates spatial/temporal memory
3. Episodic records accumulate during the task
4. Consolidation turns repeated episodes into semantic rules

## Live Smoke Memory Tiers
Live smoke now reports three distinct tiers:

- `sensor_smoke_pass`
  - D455 mount/init and frame/pose ingress succeeded
- `pipeline_smoke_pass`
  - frame reached the perception pipeline
  - detector may return an empty batch
- `memory_smoke_pass`
  - memory ingest/update ran with at least one observation

This distinction matters because an empty scene should not be reported as “sensor broken”.

## Empty Detection Interpretation
When diagnostics show:
- `frame_received=true`
- `detection_attempted=true`
- `detections_nonempty=false`
- `memory_update_called=false`

the meaning is:
- sensor ingress is healthy
- perception path ran
- nothing detectable reached memory update

That is a valid sensor/pipeline pass and only a memory-tier miss.

## Consolidation Loop
1. Episode ends
2. `SemanticConsolidationService` builds structured summaries
3. `SemanticMemoryStore` updates rule-like hints
4. `WorkingMemory` and planners reuse those hints on future recall/follow recovery

## Current Limits
- Semantic consolidation remains template/rule based.
- Re-id remains lightweight heuristic scoring.
- Live smoke validates ingress and memory wiring, not detector tuning quality.
