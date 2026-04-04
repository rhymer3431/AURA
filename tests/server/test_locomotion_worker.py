from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control.navdp_follower import NavDPFollowerResult
from ipc.messages import ActionCommand
from locomotion.worker import LocomotionWorker
from runtime.planning_session import ExecutionObservation, PlannerStats, TrajectoryUpdate


def _args(*, use_navdp_follower: bool = False) -> Namespace:
    return Namespace(
        cmd_max_vx=0.5,
        cmd_max_vy=0.3,
        cmd_max_wz=0.8,
        internvla_forward_step_m=0.5,
        internvla_turn_step_deg=30.0,
        internvla_action_timeout_s=3.0,
        lookahead_distance_m=0.6,
        heading_slowdown_rad=0.6,
        traj_stale_timeout_sec=1.5,
        cmd_accel_limit=1.0,
        cmd_yaw_accel_limit=1.5,
        traj_handoff_reset_distance_m=0.35,
        traj_handoff_reset_heading_rad=0.5,
        onnx_device="cpu",
        obstacle_defense_enabled=True,
        obstacle_stop_distance_m=0.45,
        obstacle_hold_distance_m=0.70,
        obstacle_side_bias_m=0.10,
        obstacle_min_valid_fraction=0.05,
        obstacle_min_turn_wz=0.35,
        obstacle_forward_trigger_mps=0.05,
        obstacle_slow_forward_vx_mps=0.08,
        obstacle_backoff_vx_mps=0.18,
        obstacle_lateral_nudge_vy_mps=0.12,
        obstacle_recovery_hold_sec=0.75,
        use_navdp_follower=use_navdp_follower,
    )


def _planner_command() -> ActionCommand:
    return ActionCommand(action_type="LOCAL_SEARCH", task_id="planner_managed", metadata={"planner_managed": True})


def _observation(depth: np.ndarray | None = None) -> ExecutionObservation:
    depth_image = np.ones((18, 18), dtype=np.float32) if depth is None else np.asarray(depth, dtype=np.float32)
    return ExecutionObservation(
        frame_id=1,
        rgb=np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8),
        depth=depth_image,
        sensor_meta={},
        cam_pos=np.asarray([0.0, 0.0, 1.2], dtype=np.float32),
        cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
    )


class _FakeFollower:
    def __init__(self, command: np.ndarray | None = None, *, error: Exception | None = None) -> None:
        self.command = np.asarray([0.2, -0.1, 0.05] if command is None else command, dtype=np.float32)
        self.error = error
        self.calls: list[dict[str, np.ndarray]] = []
        self.closed = False

    def compute_command(
        self,
        *,
        pose_command_b: np.ndarray,
        base_lin_vel_w: np.ndarray,
        base_ang_vel_w: np.ndarray,
        robot_quat_wxyz: np.ndarray,
    ) -> NavDPFollowerResult:
        self.calls.append(
            {
                "pose_command_b": np.asarray(pose_command_b, dtype=np.float32).copy(),
                "base_lin_vel_w": np.asarray(base_lin_vel_w, dtype=np.float32).copy(),
                "base_ang_vel_w": np.asarray(base_ang_vel_w, dtype=np.float32).copy(),
                "robot_quat_wxyz": np.asarray(robot_quat_wxyz, dtype=np.float32).copy(),
            }
        )
        if self.error is not None:
            raise self.error
        return NavDPFollowerResult(command=self.command.copy(), observation=np.zeros(13, dtype=np.float32))

    def close(self) -> None:
        self.closed = True


def _trajectory_update(
    *,
    trajectory_world: np.ndarray,
    plan_version: int,
    planner_control_mode: str | None = None,
    planner_control_version: int = -1,
    planner_yaw_delta_rad: float | None = None,
    goal_version: int = -1,
    planner_control_queue: tuple[str, ...] = (),
    planner_control_progress: float = 0.0,
    stale_hold_reason: str = "",
    planner_control_reason: str = "",
) -> TrajectoryUpdate:
    return TrajectoryUpdate(
        trajectory_world=np.asarray(trajectory_world, dtype=np.float32),
        plan_version=plan_version,
        stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.0, last_plan_step=1),
        source_frame_id=1,
        stop=False,
        planner_control_mode=planner_control_mode,
        planner_control_version=planner_control_version,
        planner_yaw_delta_rad=planner_yaw_delta_rad,
        planner_control_queue=planner_control_queue,
        planner_control_progress=planner_control_progress,
        stale_hold_reason=stale_hold_reason,
        goal_version=goal_version,
        planner_control_reason=planner_control_reason,
    )


def test_locomotion_worker_defaults_to_legacy_tracker_when_navdp_follower_is_disabled() -> None:
    follower = _FakeFollower(command=np.asarray([0.21, -0.11, 0.07], dtype=np.float32))
    worker = LocomotionWorker(_args(), follower=follower)
    proposal = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.asarray([[0.8, 0.0, 0.0], [1.2, 0.4, 0.0]], dtype=np.float32),
            plan_version=5,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        robot_ang_vel_world=np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert proposal.metadata["trajectory_command_source"] == "legacy_tracker"
    assert proposal.metadata["trajectory_fallback_reason"] == "navdp_follower_disabled"
    assert proposal.metadata["planner_control_mode"] is None
    assert proposal.metadata["planner_control_queue"] == []
    assert proposal.metadata["stale_hold_reason"] == ""
    assert proposal.metadata["locomotion_state_label"] == "tracking"
    assert follower.calls == []


def test_locomotion_worker_uses_navdp_follower_in_trajectory_mode_when_opted_in() -> None:
    follower = _FakeFollower(command=np.asarray([0.21, -0.11, 0.07], dtype=np.float32))
    worker = LocomotionWorker(_args(use_navdp_follower=True), follower=follower)
    proposal = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.asarray([[0.8, 0.0, 0.0], [1.2, 0.4, 0.0]], dtype=np.float32),
            plan_version=5,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        robot_ang_vel_world=np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(proposal.command_vector, np.asarray([0.21, -0.11, 0.07], dtype=np.float32))
    assert proposal.metadata["trajectory_command_source"] == "navdp_follower"
    assert proposal.metadata["planner_control_mode"] is None
    assert proposal.metadata["locomotion_state_label"] == "tracking"
    assert proposal.evaluation.reached_goal is False
    assert not hasattr(proposal, "status")
    assert len(follower.calls) == 1
    call = follower.calls[0]
    np.testing.assert_allclose(call["base_lin_vel_w"], np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(call["base_ang_vel_w"], np.asarray([0.4, 0.5, 0.6], dtype=np.float32))
    np.testing.assert_allclose(call["robot_quat_wxyz"], np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(call["pose_command_b"][:2], np.asarray([1.2, 0.4], dtype=np.float32), atol=1.0e-4)
    assert call["pose_command_b"][3] > 0.0


def test_locomotion_worker_falls_back_to_legacy_tracker_when_follower_inference_fails() -> None:
    follower = _FakeFollower(error=RuntimeError("inference failed"))
    worker = LocomotionWorker(_args(use_navdp_follower=True), follower=follower)
    proposal = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            plan_version=6,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(proposal.command_vector, np.asarray([0.5, 0.0, 0.0], dtype=np.float32), atol=1.0e-4)
    assert proposal.metadata["trajectory_command_source"] == "legacy_tracker"
    assert "inference failed" in str(proposal.metadata["trajectory_fallback_reason"])
    assert proposal.metadata["locomotion_state_label"] == "tracking"
    assert len(follower.calls) == 1


def test_locomotion_worker_caches_follower_init_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    init_calls = {"count": 0}

    class _RaisingFollower:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            init_calls["count"] += 1
            raise RuntimeError("init failed")

    monkeypatch.setattr("locomotion.worker.NavDPFollower", _RaisingFollower)
    worker = LocomotionWorker(_args(use_navdp_follower=True))

    for plan_version in (7, 8):
        proposal = worker.execute(
            frame_idx=1,
            observation=_observation(),
            action_command=_planner_command(),
            trajectory_update=_trajectory_update(
                trajectory_world=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                plan_version=plan_version,
            ),
            robot_pos_world=np.zeros(3, dtype=np.float32),
            robot_lin_vel_world=np.zeros(3, dtype=np.float32),
            robot_ang_vel_world=np.zeros(3, dtype=np.float32),
            robot_yaw=0.0,
            robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        assert proposal.metadata["trajectory_command_source"] == "legacy_tracker"
        assert "init failed" in str(proposal.metadata["trajectory_fallback_reason"])
        assert proposal.metadata["locomotion_state_label"] == "tracking"

    assert init_calls["count"] == 1


def test_locomotion_worker_non_trajectory_modes_skip_follower() -> None:
    follower = _FakeFollower()
    worker = LocomotionWorker(_args(use_navdp_follower=True), follower=follower)

    look_at = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=ActionCommand(action_type="LOOK_AT", task_id="look", look_at_yaw_rad=np.pi / 2.0),
        trajectory_update=_trajectory_update(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=10,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert float(look_at.command_vector[2]) > 0.0
    assert look_at.metadata["locomotion_state_label"] == "look-at"

    yaw_delta = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=11,
            planner_control_mode="yaw_delta",
            planner_yaw_delta_rad=0.4,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert float(yaw_delta.command_vector[2]) > 0.0
    assert yaw_delta.metadata["locomotion_state_label"] == "yaw-delta"

    stop = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=12,
            planner_control_mode="stop",
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(stop.command_vector, np.zeros(3, dtype=np.float32))
    assert stop.metadata["locomotion_state_label"] == "waiting"

    wait = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=13,
            planner_control_mode="wait",
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(wait.command_vector, np.zeros(3, dtype=np.float32))
    assert wait.metadata["locomotion_state_label"] == "waiting"

    assert follower.calls == []


def test_locomotion_worker_arms_direct_turns_from_planner_control_version() -> None:
    worker = LocomotionWorker(_args(use_navdp_follower=True), follower=_FakeFollower())

    yaw_left = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=0,
            planner_control_mode="yaw_left",
            planner_control_version=0,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert float(yaw_left.command_vector[2]) > 0.0
    assert yaw_left.metadata["locomotion_state_label"] == "yaw-left-override"

    yaw_right = worker.execute(
        frame_idx=2,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=0,
            planner_control_mode="yaw_right",
            planner_control_version=1,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert float(yaw_right.command_vector[2]) < 0.0
    assert yaw_right.metadata["locomotion_state_label"] == "yaw-right-override"


def test_locomotion_worker_arms_forward_from_planner_control_version() -> None:
    worker = LocomotionWorker(_args(use_navdp_follower=True), follower=_FakeFollower())

    proposal = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=0,
            planner_control_mode="forward",
            planner_control_version=0,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert float(proposal.command_vector[0]) > 0.0
    assert proposal.metadata["locomotion_state_label"] == "forward-override"


def test_locomotion_worker_resets_tracker_progress_when_goal_version_changes() -> None:
    follower = _FakeFollower()
    worker = LocomotionWorker(_args(use_navdp_follower=True), follower=follower)
    loop_traj = np.asarray(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.8, 0.8, 0.0], [0.0, 0.8, 0.0], [0.0, 0.05, 0.0]],
        dtype=np.float32,
    )
    worker._tracker.set_trajectory(loop_traj, plan_version=1, timestamp=0.0, reset_progress=False, seed_progress_idx=4)  # noqa: SLF001
    worker._last_applied_plan_version = 1  # noqa: SLF001
    worker._last_goal_version = 10  # noqa: SLF001

    same_goal = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=loop_traj + np.asarray([0.0, 0.02, 0.0], dtype=np.float32),
            plan_version=2,
            goal_version=10,
        ),
        robot_pos_world=np.asarray([0.0, 0.04, 0.8], dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert worker._tracker.progress_idx == 4  # noqa: SLF001
    assert same_goal.metadata["trajectory_command_source"] == "navdp_follower"

    goal_changed = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=loop_traj + np.asarray([0.0, 0.02, 0.0], dtype=np.float32),
            plan_version=3,
            goal_version=11,
        ),
        robot_pos_world=np.asarray([0.0, 0.04, 0.8], dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert worker._tracker.progress_idx == 0  # noqa: SLF001
    assert goal_changed.metadata["trajectory_command_source"] == "navdp_follower"


def test_locomotion_worker_exposes_direct_action_queue_and_stale_hold_metadata() -> None:
    worker = LocomotionWorker(_args(use_navdp_follower=True), follower=_FakeFollower())

    direct_action = worker.execute(
        frame_idx=1,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=0,
            planner_control_mode="forward",
            planner_control_version=0,
            planner_control_queue=("yaw_left",),
            planner_control_progress=0.4,
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert direct_action.metadata["planner_control_mode"] == "forward"
    assert direct_action.metadata["planner_control_queue"] == ["yaw_left"]
    assert direct_action.metadata["planner_control_progress"] == 0.4
    assert direct_action.metadata["locomotion_state_label"] == "forward-override"

    stale_hold = worker.execute(
        frame_idx=2,
        observation=_observation(),
        action_command=_planner_command(),
        trajectory_update=_trajectory_update(
            trajectory_world=np.asarray([[0.8, 0.0, 0.0], [1.2, 0.4, 0.0]], dtype=np.float32),
            plan_version=1,
            stale_hold_reason="stale_hold",
        ),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert stale_hold.metadata["stale_hold_reason"] == "stale_hold"
    assert stale_hold.metadata["locomotion_state_label"] == "stale-hold"
