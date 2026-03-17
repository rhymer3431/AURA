"""Execution module wrappers for locomotion command synthesis."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.subgoal_executor import SubgoalExecutionResult, SubgoalExecutor


@dataclass
class ExecutionModule:
    """Execution facade that preserves SubgoalExecutor as the backend."""

    executor: SubgoalExecutor

    def initialize(self, simulation_app, stage) -> None:
        self.executor.initialize(simulation_app, stage)

    def execute(self, **kwargs) -> SubgoalExecutionResult:  # noqa: ANN003
        return self.executor.step(**kwargs)

    def shutdown(self) -> None:
        self.executor.shutdown()
