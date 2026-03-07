# Memory Architecture

## Default Path
- Default runtime path is direct IPC plus external structured memory, not HTTP.
- `runtime.g1_bridge.NavDPCommandSource` is now a subgoal executor.
- `services.task_orchestrator.TaskOrchestrator` owns task semantics, memory recall, critic, and recovery.
- `runtime.planning_session.PlanningSession` owns direct in-process NavDP execution for `NAV_TO_POSE`, `NAV_TO_PLACE`, and `LOCAL_SEARCH`.

## Stores
- `SpatialMemoryStore`
  - Maintains anchored `PlaceNode` and `ObjectNode` records.
  - Every object is anchored to `last_place_id`.
  - Association uses track id, class, XY distance, time gap, and embedding id.
  - Static pose jumps set `conflict_flag` instead of deleting history.
- `TemporalMemoryStore`
  - FIFO buffer for speaker events, speaker bindings, person tracks, re-id candidates, follow loss, recovery actions, and object observations.
  - Used for lost-follow recovery without destructive forgetting.
- `EpisodicMemoryStore`
  - Stores `command_text`, `intent`, `target_json`, `visited_places`, `objects_seen`, candidate object/place ids, applied semantic rules, follow target id, speaker-bound person id, recovery actions, and summary tags.
- `SemanticMemoryStore`
  - Stores learned rules such as `find:apple:kitchen` and `follow:person:corner_loss`.
  - Tracks `trigger_signature`, `rule_type`, `planner_hint`, `support_count`, `success_rate`, and `last_updated`.
- `WorkingMemory`
  - Selects an active subset only for the current query.
  - Non-selected candidates remain stored and become inactive rather than deleted.
  - Ranking now includes `semantic_rule_bonus` and `rule_context_bonus` in addition to recency, reachability, confidence, and stale penalty.

## Online Update Flow
1. `perception.pipeline.PerceptionPipeline` runs detector -> tracker -> depth projection -> object mapper.
2. `perception.person_tracker.PersonTracker` and `perception.reid_store.ReIdStore` assign stable `person_id` values before memory update.
3. `ObsObject` instances are passed into `MemoryService.observe_objects()`.
4. `SpatialMemoryStore.associate_observation()` updates only the relevant local subgraph.
5. `TemporalMemoryStore` records speaker bindings, re-id candidates, follow loss, and object history for recovery and critic use.
6. `TaskOrchestrator` queries `MemoryQueryEngine` and `WorkingMemory` when a remembered-object task arrives.

## Live Smoke Ingress Checks
- Dedicated live smoke diagnostics do not try to prove full task quality.
- They instrument the minimum memory-facing path:
  1. D455 asset resolution and mount
  2. live RGB/depth frame ingress
  3. `IsaacObservationBatch` reconstruction parity with the normal bridge path
  4. `Supervisor.process_frame(...)`
  5. `MemoryService.observe_objects(...)`
  6. optional `MemoryService.update_from_observation(...)` when at least one observation exists
- This makes it explicit whether a failure is:
  - bootstrap-only
  - sensor-only
  - perception produced no detections
  - memory update path never ran

## Behavior Scenarios
- Attend caller
  - `speaker_event` enters temporal memory.
  - `TaskOrchestrator` emits `LOOK_AT`.
  - `AttentionService` keeps a short TTL queue and binds the event to a stable `person_id`.
- Follow target
  - `FollowService` binds `target_person_id` and keeps raw `track_id` only as the current observation handle.
  - Live person observations refresh temporal memory and emit `reid_candidate` events.
  - `RecoveryPlanner` tries:
    1. recent exact `person_id`
    2. spatially continuous re-id candidate
    3. last visible pose
    4. cone-search local search
- Recall remembered object
  - `IntentService` resolves `goto_remembered_object`.
  - `MemoryQueryEngine` recalls object candidates.
  - `WorkingMemory` ranks them with semantic rule bonus, semantic context bonus, recency, reachability, confidence, context, and stale terms.
  - `ObjectSearchService` emits `NAV_TO_PLACE` and, after arrival, `LOCAL_SEARCH`.
  - Selected semantic hints are attached to `ActionCommand.metadata`.
  - `PlanCritic` can trigger replan to the next candidate.

## Consolidation Loop
1. `MemoryService.finish_episode()` finalizes the active `EpisodeRecord`.
2. `services.semantic_consolidation.SemanticConsolidationService` generates summary tags and template-based summaries.
3. Successful and failed episodes update structured semantic rules:
   - `find:{target}:{room}` -> preferred room/place/object/support surfaces
   - `follow:person:corner_loss` -> last visible corner then cone-search recovery
4. Subsequent recall and follow recovery queries read those rules through `MemoryQueryEngine`, `WorkingMemory`, and `FollowService.recovery_semantic_hints()`.

## Detector and Perception Notes
- Preferred detector order is:
  1. `artifacts/models/yoloe-26s-seg-pf.engine`
  2. `TensorRtYoloeDetector` if TensorRT runtime and engine are usable
  3. `ColorSegFallbackDetector` otherwise
- `DetectorRuntimeReport` records engine discovery, TensorRT import, deserialize result, serialization mismatch, binding metadata, and backend selection reason.
- YOLOE bbox decode and segmentation-mask post-processing are modularized under `src/inference/detectors/postprocess/`.
- Fallback path remains fully testable and is the expected path in environments without TensorRT or with incompatible engine/runtime versions.

## Current Limits
- TensorRT runtime execution still depends on a matching engine/runtime/CUDA environment; serialization mismatch continues to fall back by design.
- Re-ID is lightweight heuristic scoring, not a full learned re-id backend.
- Semantic consolidation is template/rule based; no LLM narrative summarization is in the fast path.
- Live smoke can now prove frame ingress separately from memory update, but a no-detection scene can still leave `memory_updated` incomplete even when sensor ingress is healthy.
