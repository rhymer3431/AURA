from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ipc.messages import ActionCommand, ActionStatus
from locomotion.types import CommandEvaluation
from runtime.planning_session import TrajectoryUpdate


@dataclass(frozen=True)
class LocomotionProposal:
    command_vector: np.ndarray
    trajectory_update: TrajectoryUpdate
    evaluation: CommandEvaluation
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandProposal:
    source: str
    priority: int
    action_command: ActionCommand | None = None
    command_vector: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    status: ActionStatus | None = None
    trajectory_update: TrajectoryUpdate | None = None
    evaluation: CommandEvaluation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedCommand:
    action_command: ActionCommand | None
    command_vector: np.ndarray
    trajectory_update: TrajectoryUpdate
    evaluation: CommandEvaluation
    status: ActionStatus | None
    source: str
    safety_override: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
