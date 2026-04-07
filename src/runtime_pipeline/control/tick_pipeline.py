from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from server.main_control_server import ServerTickResult


@dataclass(slots=True)
class TickPipeline:
    task_manager: object
    planner_coordinator: object
    decision_engine: object
    world_state_port: object
    enrichment_service: object
    planning_service: object
    safety_resolution_service: object

    def run(
        self,
        *,
        frame_event,
        task_events,
        runtime_status,
        robot_pos_world,
        robot_lin_vel_world,
        robot_ang_vel_world,
        robot_yaw: float,
        robot_quat_wxyz,
    ) -> "ServerTickResult":  # noqa: ANN001
        from server.main_control_server import ServerTickResult

        now_ns = time.time_ns()
        notices: list[object] = []

        # 1. Drain task/runtime-control events
        initial_task = self.task_manager.snapshot()
        for event in task_events:
            notices.extend(
                self.task_manager.handle_event(
                    event,
                    planner_coordinator=self.planner_coordinator,
                    memory_client=self.safety_resolution_service.memory_client,
                )
            )

        # 2. Update mode/task in canonical state
        self.planner_coordinator.set_execution_mode(self.task_manager.mode)
        self.world_state_port.set_mode(self.task_manager.mode)
        self.world_state_port.update_task(self.task_manager.snapshot())

        # 3. Seed planning state
        self.world_state_port.seed_planning_state(
            mode=self.task_manager.mode,
            instruction=self.task_manager.snapshot().instruction,
            route_state=self.task_manager.route_state_seed(),
        )

        # 4. If task changed, reset recovery
        if self.task_manager.snapshot() != initial_task:
            self.world_state_port.reset_recovery_state(entered_at_ns=now_ns, reason="task_reset")

        # 5. Ingest new frame into canonical state
        self.world_state_port.ingest_frame(frame_event)
        current_world_state = self.world_state_port.snapshot()
        current_task = self.task_manager.snapshot()
        current_recovery = current_world_state.recovery_state
        active_memory_instruction = self.planner_coordinator.active_memory_instruction()

        # 6. Compute decision directive
        directive = self.decision_engine.evaluate(
            world_state=current_world_state,
            task=current_task,
            frame_event=frame_event,
            manual_command_present=self.task_manager.manual_command() is not None,
            active_memory_instruction=active_memory_instruction,
            recovery_state=current_recovery,
            now_ns=now_ns,
        )

        # 7-8. Enrich observation and record perception/memory into canonical state
        enrichment = self.enrichment_service.enrich(
            frame_event=frame_event,
            retrieve_memory=directive.retrieve_memory,
            instruction=active_memory_instruction,
            task=current_task,
        )
        enriched_world_state = self.world_state_port.snapshot()

        # 9-10. Optionally route through task-command flow and resolve the initial proposal
        prepared = self.safety_resolution_service.prepare_proposal(
            frame_event=frame_event,
            directive=directive,
            runtime_status=runtime_status,
            manual_command=self.task_manager.manual_command(),
        )

        # 11-13. Build planning context, execute-or-skip, and evaluate the planning outcome
        planning = self.planning_service.plan(
            frame_event=frame_event,
            world_state=current_world_state,
            task=current_task,
            recovery_state=current_recovery,
            now_ns=now_ns,
            instruction=active_memory_instruction,
            planner_mode=self.task_manager.mode,
            perception_summary=dict(enriched_world_state.perception.summary),
            memory_summary=dict(enriched_world_state.memory.summary),
            action_command=prepared.proposal.action_command,
            allow_planning=directive.allow_planning,
            skip_reason=str(directive.reason),
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_lin_vel_world=np.asarray(robot_lin_vel_world, dtype=np.float32),
            robot_ang_vel_world=np.asarray(robot_ang_vel_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
            robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
            observation=enrichment.observation,
        )

        # 14-20. Evaluate safety, update recovery, resolve execution, sync lifecycle, and persist decisions
        safety_result = self.safety_resolution_service.resolve(
            frame_event=frame_event,
            execution=planning.execution,
            proposal=prepared.proposal,
            planning_evaluation=planning.planning_evaluation,
            now_ns=now_ns,
        )

        # 21. Merge viewer overlay and return the tick result
        frame_header = None if enrichment.enriched_batch is None else enrichment.enriched_batch.frame_header
        capture_report = {} if enrichment.enriched_batch is None else dict(enrichment.enriched_batch.capture_report)
        viewer_overlay: dict[str, object] = {}
        if frame_header is not None:
            raw_overlay = frame_header.metadata.get("viewer_overlay", {})
            if isinstance(raw_overlay, dict):
                viewer_overlay = dict(raw_overlay)
        viewer_overlay.update(self.planner_coordinator.viewer_overlay_state())

        return ServerTickResult(
            planning_context=planning.planning_context,
            world_state=self.world_state_port.snapshot(),
            action_command=safety_result.resolved.action_command,
            command_vector=np.asarray(safety_result.resolved.command_vector, dtype=np.float32).copy(),
            trajectory_update=safety_result.resolved.trajectory_update,
            evaluation=safety_result.resolved.evaluation,
            status=safety_result.resolved.status,
            frame_header=frame_header,
            capture_report=capture_report,
            sensor_meta=dict(frame_event.sensor_meta),
            viewer_overlay=viewer_overlay,
            notices=tuple(notices),
        )
