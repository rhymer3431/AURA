"""Shared runtime camera pitch service for standalone G1 modes."""

from __future__ import annotations

from .api import CameraPitchApiServer
from .sensor import G1NavCameraSensor
from .targeting import resolve_camera_control_prim_path


class RuntimeCameraPitchService:
    """Attach a controllable camera and expose its pitch API independent of control mode."""

    def __init__(self, args):
        self._args = args
        self._sensor: G1NavCameraSensor | None = None
        self._api_server: CameraPitchApiServer | None = None
        self._controller = None

    def bind_controller(self, controller):
        if self._controller is not None:
            return

        self._controller = controller
        camera_prim_path = resolve_camera_control_prim_path(controller.robot_prim_path, self._args.camera_prim_path)
        self._sensor = G1NavCameraSensor(
            prim_path=camera_prim_path,
            resolution=(self._args.camera_width, self._args.camera_height),
            translation=tuple(self._args.camera_pos),
            orientation_wxyz=tuple(self._args.camera_quat),
            clipping_range=(self._args.camera_near, self._args.camera_far),
            initial_pitch_deg=self._args.camera_pitch_deg,
            pitch_limits_deg=(self._args.camera_pitch_min_deg, self._args.camera_pitch_max_deg),
            attach_streams=False,
            annotator_device="cpu",
        )
        self._sensor.attach()
        self._api_server = CameraPitchApiServer(
            host=self._args.camera_api_host,
            port=self._args.camera_api_port,
            camera_sensor=self._sensor,
        )
        self._api_server.start()

    def reset(self):
        if self._sensor is not None:
            self._sensor.apply_pending_pitch()

    def step(self):
        if self._sensor is not None:
            self._sensor.apply_pending_pitch()

    def shutdown(self):
        if self._api_server is not None:
            self._api_server.shutdown()
            self._api_server = None
        if self._sensor is not None:
            self._sensor.shutdown()
            self._sensor = None
        self._controller = None
