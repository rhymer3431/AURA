from __future__ import annotations

import time
from dataclasses import dataclass

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter, IsaacBridgeAdapterConfig
from apps.runtime_common import build_frame_source, frame_sample_to_batch, infer_demo_scene
from ipc.messages import TaskRequest
from runtime.supervisor import BusCycleResult, Supervisor, SupervisorConfig


@dataclass(frozen=True)
class MemoryAgentRuntimeConfig:
    poll_interval_ms: int = 100
    health_interval_sec: float = 5.0
    persist_interval_sec: float = 15.0
    max_cycles: int = 0
    idle_exit_after_sec: float = 0.0


class MemoryAgentRuntime:
    def __init__(
        self,
        args,
        *,
        bus,
        shm_ring=None,
        supervisor: Supervisor | None = None,
    ) -> None:
        self.args = args
        self.config = MemoryAgentRuntimeConfig(
            poll_interval_ms=max(int(getattr(args, "poll_interval_ms", 100)), 1),
            health_interval_sec=max(float(getattr(args, "health_interval_sec", 5.0)), 0.5),
            persist_interval_sec=max(float(getattr(args, "persist_interval_sec", 15.0)), 0.0),
            max_cycles=max(int(getattr(args, "max_cycles", 0)), 0),
            idle_exit_after_sec=max(float(getattr(args, "idle_exit_after_sec", 0.0)), 0.0),
        )
        self.supervisor = supervisor or Supervisor(
            bus=bus,
            shm_ring=shm_ring,
            config=SupervisorConfig(
                memory_db_path=str(getattr(args, "memory_db_path", "state/memory/memory.sqlite")),
                detector_model_path=str(getattr(args, "detector_model_path", "")),
                detector_device=str(getattr(args, "detector_device", "")),
            ),
        )
        self.bridge = IsaacBridgeAdapter(bus, IsaacBridgeAdapterConfig(), shm_ring=shm_ring)
        self._frame_source = None
        self._cycle_count = 0
        self._bootstrap_done = False
        self._task_submitted = False
        self._last_activity_time = time.monotonic()
        self._last_health_time = 0.0
        self._last_persist_time = 0.0
        self._last_state_key = ""

    def initialize(self) -> None:
        self.supervisor.publish_runtime_diagnostics()
        self.supervisor.memory_service.semantic_store.remember_rule(
            str(getattr(self.args, "bootstrap_rule", "find:apple:kitchen")),
            "prioritize table-like surfaces",
            succeeded=True,
            trigger_signature=str(getattr(self.args, "bootstrap_rule", "find:apple:kitchen")),
            rule_type="bootstrap",
            planner_hint={"preferred_support_classes": ["table", "counter"]},
            metadata={"source": "bootstrap"},
        )
        self._bootstrap_done = True
        if bool(getattr(self.args, "loopback", False)) or str(getattr(self.args, "bus", "inproc")) == "inproc":
            scene = str(getattr(self.args, "scene", "")).strip() or infer_demo_scene(str(getattr(self.args, "command", "")))
            self._frame_source = build_frame_source(
                mode=str(getattr(self.args, "frame_source", "auto")),
                scene=scene,
                source_name="memory_agent_loopback",
                room_id="kitchen" if scene == "apple" else "",
                strict_live=bool(getattr(self.args, "strict_live", False)),
            )
            source_report = self._frame_source.start()
            if str(getattr(self.args, "frame_source", "auto")).lower() == "live" and source_report.status != "ready":
                raise RuntimeError(f"live frame source unavailable: {source_report.notice}")

    def close(self, *, persist: bool | None = None) -> None:
        should_persist = self._cycle_count > 1 if persist is None else bool(persist)
        if should_persist:
            self.persist_snapshot(force=True)
        if self._frame_source is not None:
            self._frame_source.close()

    def run_once(self) -> BusCycleResult:
        if not self._bootstrap_done:
            self.initialize()
        if not self._task_submitted and str(getattr(self.args, "command", "")).strip() != "":
            self.bridge.publish_task_request(TaskRequest(command_text=str(getattr(self.args, "command", ""))))
            self._task_submitted = True
        if self._frame_source is not None:
            sample = self._frame_source.read()
            if sample is not None:
                self.bridge.publish_observation_batch(frame_sample_to_batch(sample))
        result = self.supervisor.run_bus_cycle_result(now=time.time())
        self._cycle_count += 1
        self._update_activity(result)
        return result

    def run(self) -> int:
        if not self._bootstrap_done:
            self.initialize()
        self._last_health_time = 0.0
        self._last_persist_time = 0.0
        try:
            while True:
                result = self.run_once()
                self._maybe_publish_diagnostics()
                self.persist_snapshot()
                self._log_cycle(result)
                if self.config.max_cycles > 0 and self._cycle_count >= self.config.max_cycles:
                    print(f"[MEMORY_AGENT] max_cycles reached: {self._cycle_count}")
                    return 0
                if self.config.idle_exit_after_sec > 0.0 and (time.monotonic() - self._last_activity_time) >= self.config.idle_exit_after_sec:
                    print(f"[MEMORY_AGENT] idle timeout reached: {self.config.idle_exit_after_sec:.2f}s")
                    return 0
                time.sleep(self.config.poll_interval_ms / 1000.0)
        except KeyboardInterrupt:
            print("[MEMORY_AGENT] interrupted")
            return 0

    def persist_snapshot(self, *, force: bool = False) -> int | None:
        if self.supervisor.memory_service.persistence is None:
            return None
        if not force and self.config.persist_interval_sec <= 0.0:
            return None
        now = time.monotonic()
        if not force and (now - self._last_persist_time) < self.config.persist_interval_sec:
            return None
        snapshot_id = self.supervisor.memory_service.persist_snapshot()
        self._last_persist_time = now
        return snapshot_id

    def _maybe_publish_diagnostics(self) -> None:
        now = time.monotonic()
        if (now - self._last_health_time) < self.config.health_interval_sec:
            return
        self.supervisor.publish_runtime_diagnostics()
        self._last_health_time = now

    def _update_activity(self, result: BusCycleResult) -> None:
        if result.task_count > 0 or result.frame_count > 0 or result.status_count > 0 or result.command is not None:
            self._last_activity_time = time.monotonic()

    def _log_cycle(self, result: BusCycleResult) -> None:
        snapshot = self.supervisor.snapshot()
        action_type = "none" if result.command is None else result.command.action_type
        state_key = f"{snapshot['state']}|{action_type}|{result.task_count}|{result.frame_count}|{result.status_count}"
        if state_key == self._last_state_key and result.command is None and result.task_count == 0 and result.frame_count == 0 and result.status_count == 0:
            return
        self._last_state_key = state_key
        print(
            "[MEMORY_AGENT] "
            f"cycle={self._cycle_count} state={snapshot['state']} detector={snapshot['detector_backend']} "
            f"action={action_type} tasks={result.task_count} frames={result.frame_count} statuses={result.status_count}"
        )
