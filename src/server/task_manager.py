from __future__ import annotations

from ipc.messages import ActionCommand, RuntimeControlRequest, RuntimeNotice, TaskRequest
from runtime.planning_session import TrajectoryUpdate
from schemas.world_state import TaskSnapshot


class TaskManager:
    def __init__(self, args) -> None:
        self._args = args
        self.mode = str(getattr(args, "planner_mode", "interactive")).strip().lower()
        self._manual_command: ActionCommand | None = None
        self._task = TaskSnapshot(mode=self.mode)
        self._last_interactive_phase = ""
        self._last_interactive_command_id = -1

    def bootstrap(self, *, planner_coordinator, memory_client) -> list[RuntimeNotice]:  # noqa: ANN001
        notices: list[RuntimeNotice] = []
        if self.mode == "pointgoal":
            goal_x = float(self._args.goal_x if self._args.goal_x is not None else 0.0)
            goal_y = float(self._args.goal_y if self._args.goal_y is not None else 0.0)
            self._manual_command = ActionCommand(
                action_type="NAV_TO_POSE",
                task_id="pointgoal",
                target_pose_xyz=(goal_x, goal_y, 0.0),
                stop_radius_m=float(getattr(self._args, "goal_tolerance_m", 0.4)),
                metadata={"source": "aura_runtime_pointgoal"},
            )
            self._task = TaskSnapshot(task_id="pointgoal", mode=self.mode, state="active")
            return notices

        if self.mode == "dual":
            instruction = str(getattr(self._args, "instruction", "")).strip()
            if instruction != "":
                planner_coordinator.ensure_navdp_service_ready(context="dual startup")
                planner_coordinator.ensure_dual_service_ready(context="dual startup")
                planner_coordinator.start_dual_task(instruction)
                memory_client.set_planner_task(
                    instruction=instruction,
                    planner_mode="dual",
                    task_state="active",
                    task_id="dual",
                )
                self._task = TaskSnapshot(task_id="dual", instruction=instruction, mode="dual", state="active")
                self._manual_command = self._planner_managed_command(task_id="dual", source="aura_runtime_dual")
            return notices

        if self.mode == "interactive":
            planner_coordinator.ensure_navdp_service_ready(context="interactive startup")
            planner_coordinator.activate_interactive_roaming("startup")
            self._manual_command = self._planner_managed_command(task_id="interactive", source="aura_runtime_interactive")
            self._task = TaskSnapshot(task_id="interactive", mode="interactive", state="idle")
        return notices

    def manual_command(self) -> ActionCommand | None:
        return self._manual_command

    def snapshot(self) -> TaskSnapshot:
        return self._task

    def submit_interactive_instruction(self, instruction: str, *, source: str, task_id: str, planner_coordinator, memory_client) -> tuple[int, RuntimeNotice]:  # noqa: ANN001,E501
        if self.mode != "interactive":
            raise RuntimeError("interactive instruction requires planner-mode=interactive")
        text = str(instruction).strip()
        if text == "":
            raise ValueError("interactive instruction must be non-empty")
        planner_coordinator.ensure_navdp_service_ready(context=f"interactive task ({source})")
        planner_coordinator.ensure_dual_service_ready(context=f"interactive task ({source})")
        command_id = int(planner_coordinator.submit_interactive_instruction(text))
        resolved_task_id = str(task_id or "interactive")
        memory_client.set_planner_task(
            instruction=text,
            planner_mode="interactive",
            task_state="pending",
            task_id=resolved_task_id,
            command_id=command_id,
        )
        self._task = TaskSnapshot(
            task_id=resolved_task_id,
            instruction=text,
            mode="interactive",
            state="pending",
            command_id=command_id,
        )
        return command_id, RuntimeNotice(
            component="main_control_server",
            level="info",
            notice="interactive task queued",
            details={
                "source": source,
                "taskId": resolved_task_id,
                "commandId": command_id,
                "instruction": text,
            },
        )

    def cancel_interactive_task(self, *, source: str, planner_coordinator, memory_client) -> tuple[bool, RuntimeNotice | None]:  # noqa: ANN001
        cancelled = bool(planner_coordinator.cancel_interactive_task())
        if not cancelled:
            return False, None
        memory_client.clear_planner_task(
            task_state="cancelled",
            reason=f"interactive task cancelled via {source}",
        )
        self._task = TaskSnapshot(task_id=self._task.task_id, instruction="", mode="interactive", state="cancelled")
        return True, RuntimeNotice(
            component="main_control_server",
            level="info",
            notice="interactive task cancelled",
            details={"source": source},
        )

    def handle_event(self, event, *, planner_coordinator, memory_client) -> list[RuntimeNotice]:  # noqa: ANN001
        notices: list[RuntimeNotice] = []
        if isinstance(event, TaskRequest):
            instruction = str(event.command_text).strip()
            if instruction == "":
                notices.append(
                    RuntimeNotice(
                        component="main_control_server",
                        level="warning",
                        notice="ignored empty task request",
                        details={"taskId": str(event.task_id), "source": "dashboard"},
                    )
                )
                return notices
            if self.mode != "interactive":
                notices.append(
                    RuntimeNotice(
                        component="main_control_server",
                        level="warning",
                        notice="task request rejected",
                        details={
                            "reason": "planner_mode_not_interactive",
                            "plannerMode": self.mode,
                            "taskId": str(event.task_id),
                            "instruction": instruction,
                        },
                    )
                )
                return notices
            try:
                _, notice = self.submit_interactive_instruction(
                    instruction,
                    source="dashboard",
                    task_id=str(event.task_id),
                    planner_coordinator=planner_coordinator,
                    memory_client=memory_client,
                )
            except Exception as exc:  # noqa: BLE001
                notices.append(
                    RuntimeNotice(
                        component="main_control_server",
                        level="error",
                        notice="task request failed",
                        details={
                            "taskId": str(event.task_id),
                            "instruction": instruction,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
            else:
                notices.append(notice)
            return notices

        if isinstance(event, RuntimeControlRequest):
            action = str(event.action).strip().lower()
            if action != "cancel_interactive_task":
                notices.append(
                    RuntimeNotice(
                        component="main_control_server",
                        level="warning",
                        notice="unsupported runtime control request",
                        details={"action": action},
                    )
                )
                return notices
            cancelled, notice = self.cancel_interactive_task(
                source="dashboard",
                planner_coordinator=planner_coordinator,
                memory_client=memory_client,
            )
            if not cancelled:
                notices.append(
                    RuntimeNotice(
                        component="main_control_server",
                        level="warning",
                        notice="interactive cancel ignored",
                        details={"reason": "no_active_task"},
                    )
                )
            elif notice is not None:
                notices.append(notice)
        return notices

    def sync_after_update(self, update: TrajectoryUpdate, *, memory_client) -> bool:  # noqa: ANN001
        reset_recovery = False
        if self.mode == "interactive":
            current_phase = str(update.interactive_phase or "")
            current_command_id = int(update.interactive_command_id)
            current_instruction = str(update.interactive_instruction)
            if current_phase == "task_active" and current_instruction != "":
                if self._last_interactive_phase != "task_active" or self._last_interactive_command_id != current_command_id:
                    memory_client.set_planner_task(
                        instruction=current_instruction,
                        planner_mode="interactive",
                        task_state="active",
                        task_id=self._task.task_id or "interactive",
                        command_id=current_command_id,
                    )
                self._task = TaskSnapshot(
                    task_id=self._task.task_id or "interactive",
                    instruction=current_instruction,
                    mode="interactive",
                    state="active",
                    command_id=current_command_id,
                )
            elif self._last_interactive_phase == "task_active" and current_phase == "roaming":
                clear_state = "completed" if bool(update.stop) else "idle"
                clear_reason = "interactive task complete" if bool(update.stop) else "interactive task cleared"
                if update.stats.last_error != "":
                    clear_state = "failed"
                    clear_reason = str(update.stats.last_error)
                memory_client.clear_planner_task(task_state=clear_state, reason=clear_reason)
                self._task = TaskSnapshot(task_id=self._task.task_id, instruction="", mode="interactive", state=clear_state)
                reset_recovery = True
            self._last_interactive_phase = current_phase
            self._last_interactive_command_id = current_command_id
            return reset_recovery

        if self.mode == "dual" and bool(update.stop) and update.planner_control_mode == "stop":
            memory_client.clear_planner_task(
                task_state="completed",
                reason="dual task complete",
            )
            self._task = TaskSnapshot(task_id="dual", instruction=self._task.instruction, mode="dual", state="completed")
            reset_recovery = True
        return reset_recovery

    @staticmethod
    def _planner_managed_command(*, task_id: str, source: str) -> ActionCommand:
        return ActionCommand(
            action_type="LOCAL_SEARCH",
            task_id=task_id,
            metadata={
                "source": source,
                "planner_managed": True,
            },
        )
