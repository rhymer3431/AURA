from __future__ import annotations

from server.world_state_store import WorldStateStore


class WorldStatePort:
    def __init__(self, store: WorldStateStore) -> None:
        self._store = store

    def configure_runtime(self, **kwargs) -> None:  # noqa: ANN003
        self._store.configure_runtime(**kwargs)

    def set_mode(self, mode: str) -> None:
        self._store.set_mode(mode)

    def update_task(self, task) -> None:  # noqa: ANN001
        self._store.update_task(task)

    def seed_planning_state(self, *, mode: str, instruction: str, route_state: dict[str, object] | None = None) -> None:
        self._store.seed_planning_state(mode=mode, instruction=instruction, route_state=route_state)

    def recovery_state(self):  # noqa: ANN001
        return self._store.recovery_state()

    def ingest_frame(self, frame_event) -> None:  # noqa: ANN001
        self._store.ingest_frame(frame_event)

    def record_perception(self, batch, *, summary: dict[str, object] | None = None) -> None:  # noqa: ANN001
        self._store.record_perception(batch, summary=summary)

    def record_memory_context(self, memory_context, *, summary: dict[str, object] | None = None, task=None) -> None:  # noqa: ANN001
        self._store.record_memory_context(memory_context, summary=summary, task=task)

    def record_planning_result(self, update, planner_state=None, *, recovery_state=None) -> None:  # noqa: ANN001
        self._store.record_planning_result(update, planner_state=planner_state, recovery_state=recovery_state)

    def record_command_decision(self, resolved, *, recovery_state=None) -> None:  # noqa: ANN001
        self._store.record_command_decision(resolved, recovery_state=recovery_state)

    def set_recovery_state(self, recovery_state) -> None:  # noqa: ANN001
        self._store.set_recovery_state(recovery_state)

    def reset_recovery_state(self, *, entered_at_ns: int = 0, reason: str = "") -> None:
        self._store.reset_recovery_state(entered_at_ns=entered_at_ns, reason=reason)

    def snapshot(self):  # noqa: ANN001
        return self._store.snapshot()

