from __future__ import annotations

from dataclasses import replace

import numpy as np

from ipc.messages import ActionStatus
from schemas.commands import ResolvedCommand
from schemas.events import FrameEvent


class SafetySupervisor:
    def __init__(self, args) -> None:
        self._safety_timeout_sec = float(getattr(args, "safety_timeout_sec", 20.0))
        self._traj_max_stale_sec = float(getattr(args, "traj_max_stale_sec", 4.0))

    def apply(self, *, frame_event: FrameEvent, resolved: ResolvedCommand) -> ResolvedCommand:
        should_override = frame_event.observation is None
        safety_reason = "sensor_unavailable" if should_override else ""
        stale_sec = float(resolved.trajectory_update.stale_sec)
        if stale_sec > self._traj_max_stale_sec > 0.0:
            should_override = True
            safety_reason = "trajectory_stale"

        frame_age_sec = max((float(frame_event.metadata.timestamp_ns) - float(frame_event.timestamp_ns)) / 1.0e9, 0.0)
        if frame_age_sec > self._safety_timeout_sec > 0.0:
            should_override = True
            safety_reason = "timeout"

        if not should_override:
            return resolved

        status = resolved.status
        if status is not None and status.state == "running":
            status = replace(status, state="stale", success=False, reason="safety override")
        elif status is None and resolved.action_command is not None:
            status = ActionStatus(
                command_id=str(resolved.action_command.command_id),
                state="stale",
                success=False,
                reason="safety override",
            )
        return replace(
            resolved,
            command_vector=np.zeros(3, dtype=np.float32),
            status=status,
            safety_override=True,
            metadata={**dict(resolved.metadata), "safety_override": True, "safety_reason": safety_reason},
        )
