# Dual-System Pipeline

## Default Scope
- This document explains the Isaac Sim frame path used by `scripts/powershell/run_pipeline.ps1` and the code-level dual-system planner path behind it.
- It focuses on `runtime.g1_bridge`, not live smoke.
- "System 2" means the VLM-based pixel-goal selector.
- "System 1" means the NavDP-based trajectory planner that consumes the System 2 pixel goal.

## Current Launcher Status
- Canonical launcher: `scripts/powershell/run_pipeline.ps1`
- Current launcher-exposed planner modes:
  - `interactive`
  - `pointgoal`
- `planner-mode=dual` still exists in code, but is not accepted by the current PowerShell launcher.
- In practice:
  - `interactive` starts in no-goal roaming.
  - The dual-system path becomes active after the user submits a natural-language task from the same terminal.

## Main Modules
- `scripts/powershell/run_pipeline.ps1`
  - launches Isaac Python with `python -m runtime.g1_bridge`
- `src/runtime/g1_bridge.py`
  - owns the per-frame bridge loop
  - captures observations
  - feeds perception/memory
  - feeds planning/subgoal execution
- `src/runtime/planning_session.py`
  - owns planner-mode specific logic
  - selects `pointgoal`, `nogoal`, or `dual`
- `src/services/dual_orchestrator.py`
  - dual server core
  - runs System 2 and System 1 with cache/TTL rules
- `src/runtime/subgoal_executor.py`
  - converts trajectory output into low-level motion commands
- `src/runtime/supervisor.py`
  - runs perception -> memory/orchestration side

## One Isaac Frame, Two Consumers
Each Isaac frame is captured once and then consumed by two parallel paths:

1. Perception and memory path
2. Planning and locomotion path

This split happens in `NavDPCommandSource.update()`.

### Shared Observation Payload
The planning-side observation contains:
- `rgb`
- `depth`
- `sensor_meta`
- `cam_pos`
- `cam_quat`
- `intrinsic`

The supervisor-side batch contains:
- `FrameHeader`
- `robot_pose_xyz`
- `robot_yaw_rad`
- `rgb_image`
- `depth_image_m`
- `camera_intrinsic`
- metadata overlays

## Path A: Perception and Memory
Flow:

1. `PlanningSession.capture_observation()` captures RGB, depth, pose, and intrinsics.
2. `runtime.g1_bridge` wraps the same frame into `IsaacObservationBatch`.
3. `Supervisor.process_frame()` sends the frame through:
   - detector
   - tracker
   - depth projection
   - object mapping
   - observation fusion
4. The resulting observations are ingested into memory/orchestration.

Notes:
- This path is not System 1 or System 2.
- It is a parallel consumer of the same Isaac frame.
- In `g1_view` mode the same batch is also published over ZMQ plus shared memory for the external viewer.

## Path B: Planning and Locomotion
Planner-side flow starts from the same captured observation but then depends on planner mode.

### PointGoal Mode
Flow:

1. Launcher passes `--goal-x` and `--goal-y`.
2. `runtime.g1_bridge` builds a manual `NAV_TO_POSE` command.
3. `PlanningSession` converts the world goal into robot-frame pointgoal input.
4. `AsyncPointGoalPlanner` calls NavDP directly.
5. Returned trajectory is converted to world coordinates.
6. `SubgoalExecutor` converts the trajectory into `vx`, `vy`, `wz`.

This path does not use System 2.

### Interactive Mode: Roaming Phase
Interactive mode has two internal phases:

- `roaming`
- `task_active`

Roaming flow:

1. Runtime starts with a planner-managed `LOCAL_SEARCH` command.
2. `PlanningSession._update_interactive_roaming()` submits `NoGoalPlannerInput`.
3. `AsyncNoGoalPlanner` calls NavDP in no-goal mode.
4. Returned trajectory is used for local roaming motion.

Roaming uses neither System 2 nor the dual server.

### Interactive Mode: Task Phase
The dual-system path begins only after the user enters a natural-language instruction.

Flow:

1. User enters text in the terminal.
2. `runtime.g1_bridge` checks:
   - NavDP server health
   - dual server health
3. `PlanningSession.submit_interactive_instruction()` queues the instruction.
4. `PlanningSession._activate_task()` sends `dual_reset(...)` to the dual server.
5. Subsequent frames use the dual planner path.

## System 2
System 2 is the VLM-driven goal selector.

### Inputs
- current RGB frame
- current task instruction
- event flags
  - `force_s2`
  - `stuck`
  - `collision_risk`
- image width and height

### Output Contract
System 2 must return JSON with:
- `pixel_x`
- `pixel_y`
- `stop`
- `reason`

Meaning:
- `pixel_x`, `pixel_y`
  - image-space target chosen by the VLM
- `stop`
  - whether the robot should remain stopped now
- `reason`
  - short explanation of the choice

### System 2 Behavior
- Output is cached as `GoalCache`.
- If the first System 2 result says `stop=true` before any confirmed trajectory exists, that stop is suppressed.
- Repeated identical goals can keep the same `goal_version`.
- System 2 can run in:
  - `llm` mode
  - `mock` mode

## System 1
System 1 is the NavDP planner that consumes the System 2 pixel goal.

### Inputs
- RGB frame
- depth frame
- `pixel_goal`
- `sensor_meta`
- `cam_pos`
- `cam_quat_wxyz`

### Transport
- `PlanningSession` sends the frame to the dual server through `DualSystemClient.dual_step(...)`.
- The dual server internally calls NavDP `/pixelgoal_step`.

### Output
- `trajectory_world`
- `traj_version`
- latency and debug state

Meaning:
- `trajectory_world`
  - world-coordinate path that the low-level tracker can follow

### System 1 Behavior
- System 1 output is cached as `TrajectoryCache`.
- If the goal changed while System 1 was still computing, stale plans are dropped.
- If the cached trajectory is older than `traj_ttl_sec`, it is considered low-confidence.
- If it exceeds `traj_max_stale_sec`, it is dropped and System 2 is forced again.

## Dual-System End-to-End Flow
Task-active frame flow:

1. Isaac Sim produces RGB, depth, camera pose, and robot pose.
2. `runtime.g1_bridge` captures one observation.
3. That frame goes to the perception/memory path in parallel.
4. The same observation is submitted to `AsyncDualPlanner`.
5. `DualSystemClient.dual_step(...)` sends:
   - RGB
   - depth
   - `sensor_meta`
   - `cam_pos`
   - `cam_quat_wxyz`
   - event flags
6. `DualOrchestrator.step()` decides whether to launch:
   - System 2
   - System 1
   - both
   - neither, if cached results are still valid
7. System 2 may refresh the current pixel goal.
8. System 1 may refresh the current world trajectory for that goal.
9. The dual server returns:
   - `trajectory_world`
   - `pixel_goal`
   - `stop`
   - `goal_version`
   - `traj_version`
   - `stale_sec`
   - debug state
10. `PlanningSession` accepts the newest dual plan.
11. `SubgoalExecutor` pushes that trajectory into `TrajectoryTracker`.
12. `TrajectoryTracker` outputs low-level command vectors:
   - `vx`
   - `vy`
   - `wz`

## Timing and Cadence
Important cadence controls:
- `s1_period_sec`
  - how often System 1 may refresh trajectory
- `s2_period_sec`
  - how often System 2 may refresh goal
- `goal_ttl_sec`
  - how long a System 2 goal stays fresh
- `traj_ttl_sec`
  - how long a System 1 trajectory is fresh
- `traj_max_stale_sec`
  - hard limit before dropping cached trajectory
- `dual_request_gap_frames`
  - bridge-side frame gap before re-submitting dual work

Practical effect:
- System 2 normally runs slower.
- System 1 normally runs faster.
- The bridge can keep following a cached trajectory while waiting for fresh dual responses.

## Runtime States to Keep Distinct
- Perception pipeline
  - detector/tracker/depth projection/memory ingress
- No-goal roaming
  - planner-managed exploration without System 2
- Dual task execution
  - System 2 pixel-goal selection plus System 1 trajectory generation

These are different states even though they may consume the same frame.

## Current Limits
- The PowerShell launcher does not currently expose `planner-mode=dual` directly.
- The normal `run_pipeline.ps1` default path is not always the dual-system path.
- Dual execution in current launcher usage usually means:
  - start in `interactive`
  - submit a natural-language task
- The repository still contains legacy HTTP transport for NavDP and dual server interactions in this path.
