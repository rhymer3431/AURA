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
  - FIFO buffer for speaker events, person tracks, object observations, and critic-side reacquire hints.
  - Used for lost-follow recovery without destructive forgetting.
- `EpisodicMemoryStore`
  - Stores `command_text`, `intent`, `target_json`, `visited_places`, `objects_seen`, success/failure fields, and summaries.
- `SemanticMemoryStore`
  - Stores learned rules such as `find:apple:kitchen`.
  - Tracks support and success rate for recall scoring.
- `WorkingMemory`
  - Selects an active subset only for the current query.
  - Non-selected candidates remain stored and become inactive rather than deleted.

## Online Update Flow
1. `perception.pipeline.PerceptionPipeline` runs detector -> tracker -> depth projection -> object mapper.
2. `ObsObject` instances are passed into `MemoryService.observe_objects()`.
3. `SpatialMemoryStore.associate_observation()` updates only the relevant local subgraph.
4. `TemporalMemoryStore` records follow/speaker/object history for recovery and critic use.
5. `TaskOrchestrator` queries `MemoryQueryEngine` and `WorkingMemory` when a remembered-object task arrives.

## Behavior Scenarios
- Attend caller
  - `speaker_event` enters temporal memory.
  - `TaskOrchestrator` emits `LOOK_AT`.
  - Person observations can bind the event to a persistent track id.
- Follow target
  - `FollowService` binds `target_track_id`.
  - Live person observations refresh temporal memory.
  - `RecoveryPlanner` uses temporal history to issue a reacquire `NAV_TO_POSE`.
- Recall remembered object
  - `IntentService` resolves `goto_remembered_object`.
  - `MemoryQueryEngine` recalls object candidates.
  - `WorkingMemory` ranks them with semantic, recency, reachability, confidence, context, and stale terms.
  - `ObjectSearchService` emits `NAV_TO_PLACE` and, after arrival, `LOCAL_SEARCH`.
  - `PlanCritic` can trigger replan to the next candidate.

## Detector and Perception Notes
- Preferred detector order is:
  1. `artifacts/models/yoloe-26s-seg-pf.engine`
  2. `TensorRtYoloeDetector` if TensorRT runtime and engine are usable
  3. `ColorSegFallbackDetector` otherwise
- Current TensorRT backend performs engine discovery and load validation, but YOLOE decode/post-processing is still TODO.
- Fallback path remains fully testable and is the expected path in environments without TensorRT or with incompatible engine/runtime versions.

## Current Limits
- TensorRT YOLOE decode and binding-specific post-processing are not implemented yet.
- ReID and richer multi-person speaker binding are still lightweight skeletons.
- Episodic summarization and semantic consolidation are still simple rule accumulation rather than full narrative summarization.
