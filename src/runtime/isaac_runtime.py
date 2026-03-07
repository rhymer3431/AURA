from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from adapters.sensors.isaac_live_source import IsaacLiveFrameSource, IsaacLiveSourceConfig
from apps.runtime_common import frame_sample_to_batch
from common.geometry import quat_wxyz_to_yaw
from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from ipc.messages import ActionStatus
from ipc.messages import FrameHeader

from .planning_session import PlanningSession, TrajectoryUpdate
from .supervisor import Supervisor, SupervisorConfig


@dataclass
class RuntimeStepResult:
    action_status: ActionStatus
    trajectory_update: TrajectoryUpdate


class IsaacRuntime:
    def __init__(self, args: argparse.Namespace, *, supervisor: Supervisor | None = None, planning_session: PlanningSession | None = None) -> None:
        self.args = args
        self.supervisor = supervisor or Supervisor()
        self.planning_session = planning_session or PlanningSession(args)
        self._active_command = None
        self._frame_source: IsaacLiveFrameSource | None = None
        self._latest_env = None
        self._latest_robot_pose = (0.0, 0.0, 0.0)
        self._latest_robot_yaw = 0.0

    def initialize(self, simulation_app, stage) -> None:
        self.planning_session.initialize(simulation_app, stage)
        self._frame_source = IsaacLiveFrameSource(
            sensor_adapter=self.planning_session.sensor,
            env_provider=lambda: self._latest_env,
            robot_pose_provider=lambda: self._latest_robot_pose,
            robot_yaw_provider=lambda: self._latest_robot_yaw,
            config=IsaacLiveSourceConfig(
                source_name="isaac_runtime_live",
                strict_live=bool(getattr(self.args, "strict_live", False)),
            ),
        )
        self._frame_source.start()

    def submit_task(self, command_text: str, *, target_json: dict[str, object] | None = None, speaker_id: str = "") -> None:
        self.supervisor.submit_task(command_text, target_json=target_json, speaker_id=speaker_id)

    def update(self, frame_id: int, *, robot_pos_world: np.ndarray, robot_quat_wxyz: np.ndarray, env=None) -> RuntimeStepResult:  # noqa: ANN001
        robot_pose = tuple(float(v) for v in np.asarray(robot_pos_world, dtype=np.float32).reshape(-1)[:3])
        self._latest_env = env
        self._latest_robot_pose = robot_pose
        self._latest_robot_yaw = float(quat_wxyz_to_yaw(np.asarray(robot_quat_wxyz, dtype=np.float32)))
        observation = None
        if self._frame_source is not None:
            sample = self._frame_source.read()
            if sample is not None:
                sample.frame_id = int(frame_id)
                batch = frame_sample_to_batch(sample)
                self.supervisor.process_frame(batch, publish=False)
                observation = self.planning_session.build_local_observation(
                    frame_id=int(frame_id),
                    rgb=batch.rgb_image,
                    depth=batch.depth_image_m,
                    camera_pose_xyz=batch.frame_header.camera_pose_xyz,
                    camera_quat_wxyz=batch.frame_header.camera_quat_wxyz,
                    intrinsic=batch.camera_intrinsic,
                    sensor_meta=dict(batch.frame_header.metadata),
                )
        if observation is None:
            observation = self.planning_session.capture_observation(frame_id, env=env)
            if observation is not None:
                self.supervisor.process_frame(
                    IsaacObservationBatch(
                        frame_header=FrameHeader(
                            frame_id=int(observation.frame_id),
                            timestamp_ns=time.time_ns(),
                            source="isaac_runtime",
                            width=int(observation.rgb.shape[1]),
                            height=int(observation.rgb.shape[0]),
                            camera_pose_xyz=tuple(float(v) for v in observation.cam_pos[:3]),
                            camera_quat_wxyz=tuple(float(v) for v in observation.cam_quat[:4]),
                            robot_pose_xyz=robot_pose,
                            robot_yaw_rad=float(self._latest_robot_yaw),
                            sim_time_s=float(time.time()),
                            metadata=dict(observation.sensor_meta),
                        ),
                        robot_pose_xyz=robot_pose,
                        robot_yaw_rad=float(self._latest_robot_yaw),
                        sim_time_s=float(time.time()),
                        rgb_image=observation.rgb,
                        depth_image_m=observation.depth,
                        camera_intrinsic=observation.intrinsic,
                    ),
                    publish=False,
                )
        command = self.supervisor.step(now=time.time(), robot_pose=robot_pose, action_status=None, publish=False)
        self._active_command = command
        if observation is None:
            trajectory_update = self.planning_session.update(
                frame_id,
                action_command=command,
                robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(quat_wxyz_to_yaw(np.asarray(robot_quat_wxyz, dtype=np.float32))),
                robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
                env=env,
            )
        else:
            trajectory_update = self.planning_session.plan_with_observation(
                observation,
                action_command=command,
                robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                robot_yaw=float(quat_wxyz_to_yaw(np.asarray(robot_quat_wxyz, dtype=np.float32))),
                robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
            )
        status = ActionStatus(
            command_id=command.command_id if command is not None else "",
            state="running" if command is not None else "idle",
            success=False,
            robot_pose_xyz=robot_pose,
        )
        return RuntimeStepResult(action_status=status, trajectory_update=trajectory_update)

    def shutdown(self) -> None:
        if self._frame_source is not None:
            self._frame_source.close()
        self.planning_session.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Direct Isaac runtime scaffold.")
    parser.add_argument("--command", type=str, default="")
    parser.add_argument("--memory-db-path", type=str, default="state/memory/memory.sqlite")
    args = parser.parse_args(argv)
    runtime = IsaacRuntime(args, supervisor=Supervisor(config=SupervisorConfig(memory_db_path=args.memory_db_path)))
    if str(args.command).strip() != "":
        runtime.submit_task(str(args.command))
    print(f"[ISAAC_RUNTIME] configured memory_db_path={args.memory_db_path}")
    return 0
