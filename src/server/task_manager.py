from __future__ import annotations

from systems.transport.messages import ActionCommand, RuntimeControlRequest, RuntimeNotice, TaskRequest
from runtime.planning_session import TrajectoryUpdate
from schemas.execution_mode import ExecutionMode, normalize_execution_mode
from schemas.world_state import TaskSnapshot

from .execution_mode_classifier import ExecutionModeClassifier

class TaskManager:
    def __init__(self, args) -> None:
        self._args = args
        self._classifier = ExecutionModeClassifier()
        self.mode: ExecutionMode = "IDLE"
        self._manual_command: ActionCommand | None = None
        self._task = TaskSnapshot(mode=self.mode)
        self._route_state_seed: dict[str, object] = {}

    def bootstrap(self, *, planner_coordinator, memory_client) -> list[RuntimeNotice]:  # noqa: ANN001
        instruction = str(getattr(self._args, "instruction", "")).strip()
        raw_mode = str(getattr(self._args, "planner_mode", "")).strip().lower()
        normalized_mode = normalize_execution_mode(raw_mode)

        if raw_mode == "interactive":
            planner_coordinator.ensure_navdp_service_ready(context="interactive startup")
            planner_coordinator.activate_interactive_roaming("startup")
            planner_coordinator.set_execution_mode("NAV")
            self.mode = "NAV"
            self._manual_command = self._planner_managed_command(task_id="interactive", source="startup:interactive")
            self._manual_command.metadata.update({"execution_mode": "NAV", "interactive": True})
            self._route_state_seed = {"policy": "interactive_roaming"}
            self._task = TaskSnapshot(task_id="interactive", instruction="", mode="NAV", state="active", command_id=-1)
            memory_client.set_planner_task(
                instruction="",
                planner_mode="interactive",
                task_state="active",
                task_id="interactive",
                command_id=-1,
            )
            return []

        if normalized_mode == "NAV" and instruction != "":
            startup_label = raw_mode if raw_mode in {"nav"} else "nav"
            planner_coordinator.ensure_navdp_service_ready(context=f"{startup_label} startup")
            planner_coordinator.ensure_system2_service_ready(context=f"{startup_label} startup")
            planner_coordinator.start_nav_task(instruction, mode="NAV")
            planner_coordinator.set_execution_mode("NAV")
            self.mode = "NAV"
            self._manual_command = self._planner_managed_command(task_id="startup", source=f"startup:{startup_label}")
            self._manual_command.metadata["execution_mode"] = "NAV"
            self._route_state_seed = {}
            self._task = TaskSnapshot(task_id="startup", instruction=instruction, mode="NAV", state="active", command_id=-1)
            memory_client.set_planner_task(
                instruction=instruction,
                planner_mode="nav",
                task_state="active",
                task_id="startup",
                command_id=-1,
            )
            return []

        planner_coordinator.activate_idle("startup")
        self._set_idle_state()
        return []

    def manual_command(self) -> ActionCommand | None:
        return self._manual_command

    def snapshot(self) -> TaskSnapshot:
        return self._task

    def route_state_seed(self) -> dict[str, object]:
        return dict(self._route_state_seed)

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
            try:
                notices.extend(
                    self._activate_task_request(
                        event,
                        source="dashboard",
                        planner_coordinator=planner_coordinator,
                        memory_client=memory_client,
                    )
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
            return notices

        if isinstance(event, RuntimeControlRequest):
            action = str(event.action).strip().lower()
            if action not in {"set_idle", "cancel_interactive_task"}:
                notices.append(
                    RuntimeNotice(
                        component="main_control_server",
                        level="warning",
                        notice="unsupported runtime control request",
                        details={"action": action},
                    )
                )
                return notices
            changed, notice = self.set_idle(
                source="dashboard",
                planner_coordinator=planner_coordinator,
                memory_client=memory_client,
            )
            if not changed:
                notices.append(
                    RuntimeNotice(
                        component="main_control_server",
                        level="warning",
                        notice="set idle ignored",
                        details={"reason": "no_active_task"},
                    )
                )
            elif notice is not None:
                notices.append(notice)
        return notices

    def sync_after_update(self, update: TrajectoryUpdate, status, *, memory_client) -> bool:  # noqa: ANN001
        if self.mode == "IDLE":
            return False
        if self.mode == "TALK":
            memory_client.clear_planner_task(task_state="completed", reason="talk hook emitted")
            self._set_idle_state()
            return True
        terminal_state = "" if status is None else str(getattr(status, "state", "") or "")
        if update.stats.last_error != "" and terminal_state == "":
            terminal_state = "failed"
        if terminal_state not in {"succeeded", "failed"}:
            self._task = TaskSnapshot(
                task_id=self._task.task_id,
                instruction=self._task.instruction,
                mode=self.mode,
                state="active",
                command_id=self._task.command_id,
            )
            return False
        clear_state = "completed" if terminal_state == "succeeded" else "failed"
        clear_reason = str(update.stats.last_error or getattr(status, "reason", "") or clear_state)
        memory_client.clear_planner_task(task_state=clear_state, reason=clear_reason)
        self._set_idle_state()
        return True

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

    def set_idle(self, *, source: str, planner_coordinator, memory_client) -> tuple[bool, RuntimeNotice | None]:  # noqa: ANN001
        if self.mode == "IDLE" and self._task.state == "idle":
            return False, None
        memory_client.clear_planner_task(
            task_state="cancelled",
            reason=f"set idle via {source}",
        )
        planner_coordinator.activate_idle(f"set_idle:{source}")
        self._set_idle_state()
        return True, RuntimeNotice(
            component="main_control_server",
            level="info",
            notice="execution mode set to idle",
            details={"source": source, "action": "set_idle"},
        )

    def _activate_task_request(self, request: TaskRequest, *, source: str, planner_coordinator, memory_client) -> list[RuntimeNotice]:  # noqa: ANN001
        instruction = str(request.command_text).strip()
        classification = self._classifier.classify(request)
        task_id = str(request.task_id)
        planner_coordinator.activate_idle(f"switch_to_{classification.mode.lower()}")
        notices: list[RuntimeNotice] = [
            RuntimeNotice(
                component="main_control_server",
                level="info",
                notice="task classified",
                details={
                    "taskId": task_id,
                    "instruction": instruction,
                    "executionMode": classification.mode,
                    "reason": classification.reason,
                    "intent": classification.intent_name,
                },
            )
        ]
        if classification.mode == "TALK":
            self.mode = "TALK"
            self._manual_command = None
            self._route_state_seed = {"hookStatus": "ready"}
            self._task = TaskSnapshot(task_id=task_id, instruction=instruction, mode="TALK", state="active", command_id=-1)
            memory_client.set_planner_task(
                instruction=instruction,
                planner_mode="talk",
                task_state="active",
                task_id=task_id,
                command_id=-1,
            )
            return notices

        if classification.mode == "NAV":
            planner_coordinator.ensure_navdp_service_ready(context=f"NAV task ({source})")
            planner_coordinator.ensure_system2_service_ready(context=f"NAV task ({source})")
            planner_coordinator.start_nav_task(instruction, mode="NAV")
            planner_coordinator.set_execution_mode("NAV")
            self.mode = "NAV"
            self._manual_command = self._planner_managed_command(task_id=task_id, source=f"{source}:NAV")
            self._manual_command.metadata["execution_mode"] = "NAV"
            self._route_state_seed = {}
        elif classification.mode == "EXPLORE":
            planner_coordinator.ensure_navdp_service_ready(context=f"EXPLORE task ({source})")
            planner_coordinator.set_execution_mode("EXPLORE")
            self.mode = "EXPLORE"
            self._manual_command = self._planner_managed_command(task_id=task_id, source=f"{source}:EXPLORE")
            self._manual_command.metadata.update({"execution_mode": "EXPLORE", "policy": "nogoal"})
            self._route_state_seed = {"policy": "nogoal"}
        elif classification.mode == "MEM_NAV":
            planner_coordinator.ensure_navdp_service_ready(context=f"MEM_NAV task ({source})")
            target = memory_client.resolve_navigation_target(
                instruction=instruction,
                current_pose=None,
                target_class=classification.target_class,
                room_id=str(request.target_json.get("room_id", "")),
            )
            if target is None:
                self._set_idle_state()
                notices.append(
                    RuntimeNotice(
                        component="main_control_server",
                        level="error",
                        notice="memory navigation target not found",
                        details={"taskId": task_id, "instruction": instruction},
                    )
                )
                return notices
            planner_coordinator.set_execution_mode("MEM_NAV")
            self.mode = "MEM_NAV"
            self._manual_command = ActionCommand(
                action_type="NAV_TO_POSE",
                task_id=task_id,
                target_pose_xyz=target.goal_pose_xyz,
                stop_radius_m=float(getattr(self._args, "goal_tolerance_m", 0.4)),
                metadata={
                    "source": f"{source}:MEM_NAV",
                    "planner_managed": True,
                    "execution_mode": "MEM_NAV",
                    "pose_source": "memory",
                    "memory_pose_xyz": list(target.memory_pose_xyz),
                    "goal_pose_xyz": list(target.goal_pose_xyz),
                    "target_object_id": target.object_id,
                    "target_place_id": target.place_id,
                    "target_class": classification.target_class,
                },
            )
            self._route_state_seed = {
                "memoryPoseXyz": list(target.memory_pose_xyz),
                "goalPoseXyz": list(target.goal_pose_xyz),
                "waypointIndex": 0,
                "waypointCount": 0,
            }
        else:
            self._set_idle_state()
            return notices

        memory_client.set_planner_task(
            instruction=instruction,
            planner_mode=self.mode.lower(),
            task_state="active",
            task_id=task_id,
            command_id=-1,
        )
        self._task = TaskSnapshot(task_id=task_id, instruction=instruction, mode=self.mode, state="active", command_id=-1)
        return notices

    def _set_idle_state(self) -> None:
        self.mode = "IDLE"
        self._manual_command = None
        self._route_state_seed = {}
        self._task = TaskSnapshot(mode="IDLE", state="idle")
