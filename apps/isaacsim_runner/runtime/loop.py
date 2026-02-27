from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class _LoopState:
    step_idx: int = 0
    prev_start: Optional[float] = None
    dt_window: list[float] = field(default_factory=list)
    wall_start: float = 0.0

    def tick(self, loop_start: float, cmd_dt: float) -> float:
        if self.prev_start is None:
            loop_dt = cmd_dt
        else:
            loop_dt = max(0.0, loop_start - self.prev_start)
        self.prev_start = loop_start
        return loop_dt

    def compute_timing(self, loop_end: float, loop_start: float, loop_dt: float, cmd_dt: float) -> Dict[str, Optional[float]]:
        elapsed = max(0.0, loop_end - loop_start)
        overrun_ms = max(0.0, (elapsed - cmd_dt) * 1000.0)
        self.dt_window.append(loop_dt)
        if len(self.dt_window) > 400:
            self.dt_window = self.dt_window[-400:]

        publish_hz_window = None
        if self.dt_window:
            avg_loop_dt = sum(self.dt_window[-50:]) / len(self.dt_window[-50:])
            if avg_loop_dt > 0.0:
                publish_hz_window = 1.0 / avg_loop_dt

        wall_elapsed = loop_end - self.wall_start
        rtf = None
        if wall_elapsed > 1e-6:
            rtf = (self.step_idx * cmd_dt) / wall_elapsed

        return {
            "loop_overrun_ms": float(overrun_ms),
            "publish_hz_window": publish_hz_window,
            "rtf": rtf,
        }


def detect_fall_and_slip(
    *,
    base_pose: Optional[Dict[str, float]],
    base_prev_pose: Optional[Dict[str, float]],
    loop_dt: float,
    navigate_bridge,
) -> Dict[str, Any]:
    base_roll = None
    base_pitch = None
    base_yaw = None
    base_height = None
    fall_flag = False
    slip_flag = False
    next_base_prev_pose = base_prev_pose

    if base_pose is not None:
        base_roll = float(base_pose["roll"])
        base_pitch = float(base_pose["pitch"])
        base_yaw = float(base_pose["yaw"])
        base_height = float(base_pose["height"])
        fall_flag = bool(abs(base_roll) > 0.9 or abs(base_pitch) > 0.9 or base_height < 0.22)
        if base_prev_pose is not None and loop_dt > 1e-6:
            dx = float(base_pose["x"] - base_prev_pose["x"])
            dy = float(base_pose["y"] - base_prev_pose["y"])
            base_speed_xy = math.sqrt(dx * dx + dy * dy) / loop_dt
            cmd_speed = 0.0
            if navigate_bridge is not None:
                cmd_speed = math.sqrt(float(navigate_bridge._vx) ** 2 + float(navigate_bridge._vy) ** 2)
            slip_flag = bool(
                (cmd_speed < 0.05 and base_speed_xy > 0.20)
                or (cmd_speed > 0.20 and base_speed_xy < 0.01)
            )
        next_base_prev_pose = base_pose

    return {
        "base_roll": base_roll,
        "base_pitch": base_pitch,
        "base_yaw": base_yaw,
        "base_height": base_height,
        "fall_flag": bool(fall_flag),
        "slip_flag": bool(slip_flag),
        "base_prev_pose": next_base_prev_pose,
    }


def run_simulation_loop(
    *,
    simulation_app,
    simulation_context,
    stage_obj,
    cmd_dt: float,
    now_perf: Optional[Callable[[], float]] = None,
    navigate_bridge=None,
    base_pose_path: Optional[str] = None,
    read_base_pose_fn: Optional[Callable[..., Optional[Dict[str, float]]]] = None,
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    perf_now = now_perf if now_perf is not None else time.perf_counter
    state = _LoopState(wall_start=perf_now())
    base_prev_pose: Optional[Dict[str, float]] = None

    while simulation_app.is_running():
        t_loop_start = perf_now()
        loop_dt = state.tick(t_loop_start, cmd_dt)

        if navigate_bridge is not None:
            navigate_bridge.spin_once()
            navigate_bridge.apply(stage_obj, cmd_dt)
        simulation_context.step(render=True)

        t_loop_end = perf_now()
        state.step_idx += 1
        timing = state.compute_timing(t_loop_end, t_loop_start, loop_dt, cmd_dt)

        base_pose = (
            read_base_pose_fn(stage_obj, base_pose_path)
            if (read_base_pose_fn is not None and base_pose_path)
            else None
        )
        posture = detect_fall_and_slip(
            base_pose=base_pose,
            base_prev_pose=base_prev_pose,
            loop_dt=loop_dt,
            navigate_bridge=navigate_bridge,
        )
        base_prev_pose = posture["base_prev_pose"]

        if on_step is not None:
            rec: Dict[str, Any] = {
                "event": "runner_loop",
                "step_idx": state.step_idx,
                "loop_dt": float(loop_dt),
                "loop_overrun_ms": timing["loop_overrun_ms"],
                "publish_hz_window": timing["publish_hz_window"],
                "rtf": timing["rtf"],
                "fall_flag": posture["fall_flag"],
                "slip_flag": posture["slip_flag"],
            }
            if posture["base_roll"] is not None:
                rec["base_roll"] = posture["base_roll"]
            if posture["base_pitch"] is not None:
                rec["base_pitch"] = posture["base_pitch"]
            if posture["base_yaw"] is not None:
                rec["base_yaw"] = posture["base_yaw"]
            if posture["base_height"] is not None:
                rec["base_height"] = posture["base_height"]
            on_step(rec)
