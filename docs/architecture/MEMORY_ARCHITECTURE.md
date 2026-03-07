# Memory Architecture

## Overview
- NavDP + G1 locomotion stays a low-level executor. It receives subgoals and produces trajectories.
- Structured memory lives outside NavDP in `src/memory` and is surfaced through `services.memory_service.MemoryService`.
- `services.task_orchestrator.TaskOrchestrator` consumes memory queries, speaker events, and perception updates to choose the next subgoal.

## Memory Stores
- `SpatialMemoryStore`
  - Maintains a place graph plus object graph.
  - `PlaceNode` owns pose, room, neighbors, visits, and timestamps.
  - `ObjectNode` is always anchored to `last_place_id`.
  - Association uses track id, class, pose distance, time gap, and optional embedding id.
  - Static-object pose jumps raise `conflict_flag` instead of silently rewriting history.
- `TemporalMemoryStore`
  - FIFO event buffer for recent track observations, speaker events, critic flags, and follow-target history.
  - Supports follow-target reacquire without deleting older memory.
- `EpisodicMemoryStore`
  - Records task-level execution summaries and feeds semantic consolidation.
- `SemanticMemoryStore`
  - Stores reusable rules such as `find:apple:kitchen`.
  - Tracks `support_count`, `success_count`, and `success_rate`.
- `WorkingMemory`
  - Builds an active subset for the current query.
  - Scores candidates using `semantic_match`, `recency`, `reachability`, `confidence`, `context_match`, and `stale_penalty`.
  - Non-selected candidates remain stored but inactive.

## Update Flow
1. Perception converts detections into `memory.models.ObsObject`.
2. `MemoryService.observe_objects()` calls `SpatialMemoryStore.associate_observation()` and appends temporal events.
3. `TaskOrchestrator` uses `MemoryService.recall_object()` for remembered-object tasks.
4. `ObjectSearchService` converts the best memory candidate into `NAV_TO_PLACE` or `LOCAL_SEARCH`.

## Query Flow
- Example: "아까 봤던 사과를 찾아가"
  - `IntentService` resolves `goto_remembered_object` with `target_class=apple`.
  - `MemoryQueryEngine` pulls matching objects and semantic rules.
  - `WorkingMemory` ranks candidates.
  - The best object maps to `last_place_id`, which becomes a navigation subgoal.

## Persistence
- SQLite snapshot support lives in `memory.persistence.SQLiteMemoryPersistence`.
- Default runtime path is `state/memory/memory.sqlite`.
- Current implementation persists snapshots, not the full online graph mutation log.

## Current Limits
- Perception is still skeleton-level. YOLO/ReID integration is the next step.
- Direct NavDP inference is still separate from the new memory layer. Low-level planning can continue to use legacy NavDP compatibility while task semantics move out of runtime.
