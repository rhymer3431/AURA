from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import requests

from adapters.sensors.d455_sensor import D455SensorAdapter, D455SensorAdapterConfig
from control.async_planners import AsyncNoGoalPlanner, AsyncPointGoalPlanner
from inference.navdp import InProcessNavDPClient, create_inprocess_navdp_client
from inference.vlm import AsyncSystem2Planner, System2Session, System2SessionConfig
from systems.transport.messages import ActionCommand
from memory.models import MemoryContextBundle


def _health_url(base_url: str, health_path: str = "/healthz") -> str:
    return f"{str(base_url).rstrip('/')}/{str(health_path).lstrip('/')}"


def _check_remote_service(
    base_url: str,
    *,
    timeout_sec: float,
    service_name: str,
    context: str,
    health_path: str = "/healthz",
) -> None:
    url = str(base_url).strip()
    if url == "":
        raise RuntimeError(f"{context}: missing {service_name} base URL")
    try:
        response = requests.get(_health_url(url, health_path), timeout=float(timeout_sec))
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{context}: {service_name} is unavailable at {url}. "
            f"Start the required server first. detail={type(exc).__name__}: {exc}"
        ) from exc


def _is_local_service_url(base_url: str) -> bool:
    parsed = urlparse(str(base_url).strip())
    return str(parsed.hostname or "").strip().lower() in {"127.0.0.1", "localhost", "::1"}


def _resolve_local_launcher_script(script_name: str) -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "scripts" / str(script_name),
        repo_root / str(script_name),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _start_local_launcher(script_path: Path, launcher_args: tuple[str, ...] = ()) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            *launcher_args,
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ensure_remote_service_ready(
    base_url: str,
    *,
    timeout_sec: float,
    service_name: str,
    context: str,
    launcher_script_name: str | None = None,
    launcher_args: tuple[str, ...] = (),
    launcher_processes: dict[str, subprocess.Popen[Any]] | None = None,
    startup_timeout_sec: float = 45.0,
    health_path: str = "/healthz",
) -> None:
    try:
        _check_remote_service(
            base_url,
            timeout_sec=timeout_sec,
            service_name=service_name,
            context=context,
            health_path=health_path,
        )
        return
    except RuntimeError as initial_exc:
        if launcher_script_name is None or not _is_local_service_url(base_url):
            raise
        launcher_path = _resolve_local_launcher_script(launcher_script_name)
        if launcher_path is None:
            raise

        url_key = str(base_url).strip()
        launcher_proc = launcher_processes.get(url_key) if launcher_processes is not None else None
        if launcher_proc is None or launcher_proc.poll() is not None:
            try:
                launcher_proc = _start_local_launcher(launcher_path, launcher_args)
            except Exception as launch_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"{context}: {service_name} is unavailable at {url_key}. "
                    f"Auto-start via .\\{launcher_path.name} {' '.join(launcher_args)} failed. "
                    f"detail={type(launch_exc).__name__}: {launch_exc}"
                ) from launch_exc
            if launcher_processes is not None:
                launcher_processes[url_key] = launcher_proc

        deadline = time.time() + max(float(timeout_sec), float(startup_timeout_sec))
        while time.time() < deadline:
            try:
                _check_remote_service(
                    base_url,
                    timeout_sec=timeout_sec,
                    service_name=service_name,
                    context=context,
                    health_path=health_path,
                )
                return
            except RuntimeError as retry_exc:
                if launcher_proc is not None and launcher_proc.poll() is not None:
                    raise RuntimeError(
                        f"{context}: {service_name} is unavailable at {url_key}. "
                        f"Auto-start via .\\{launcher_path.name} {' '.join(launcher_args)} exited with code {launcher_proc.poll()}. "
                        f"detail={retry_exc}"
                    ) from retry_exc
                time.sleep(0.5)

        raise RuntimeError(
            f"{context}: {service_name} is unavailable at {url_key}. "
            f"Auto-start via .\\{launcher_path.name} {' '.join(launcher_args)} did not become ready within "
            f"{int(max(float(timeout_sec), float(startup_timeout_sec)))}s. detail={initial_exc}"
        ) from initial_exc


@dataclass(frozen=True)
class PlannerStats:
    successful_calls: int = 0
    failed_calls: int = 0
    latency_ms: float = 0.0
    last_error: str = ""
    last_plan_step: int = -1


@dataclass(frozen=True)
class ExecutionObservation:
    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    sensor_meta: dict[str, Any]
    cam_pos: np.ndarray
    cam_quat: np.ndarray
    intrinsic: np.ndarray
    memory_context: MemoryContextBundle | None = None


@dataclass(frozen=True)
class TrajectoryUpdate:
    trajectory_world: np.ndarray
    plan_version: int
    stats: PlannerStats
    source_frame_id: int
    goal_local_xy: np.ndarray | None = None
    action_command: ActionCommand | None = None
    stop: bool = False
    planner_control_mode: str | None = None
    planner_control_version: int = -1
    planner_control_reason: str = ""
    planner_yaw_delta_rad: float | None = None
    planner_control_queue: tuple[str, ...] = ()
    planner_control_progress: float = 0.0
    stale_sec: float = -1.0
    stale_hold_reason: str = ""
    goal_version: int = -1
    traj_version: int = -1
    used_cached_traj: bool = False
    locomotion_state_label: str = ""
    sensor_meta: dict[str, Any] | None = None
    interactive_phase: str | None = None
    interactive_command_id: int = -1
    interactive_instruction: str = ""


class PlanningSession:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        sensor_factory: Callable[[D455SensorAdapterConfig], D455SensorAdapter] | None = None,
        navdp_client_factory: Callable[[np.ndarray, argparse.Namespace], Any] | None = None,
        system2_client_factory: Callable[[argparse.Namespace], System2Session] | None = None,
    ) -> None:
        self.args = args
        self.mode = str(getattr(args, "planner_mode", "IDLE")).strip().upper() or "IDLE"
        self.sensor_factory = sensor_factory or (lambda cfg: D455SensorAdapter(cfg))
        self.navdp_client_factory = navdp_client_factory or self._default_navdp_client_factory
        self.system2_client_factory = system2_client_factory or self._default_system2_client_factory
        self.sensor: D455SensorAdapter | None = None
        self.navdp_client: Any | None = None
        self.pointgoal_planner: AsyncPointGoalPlanner | None = None
        self.nogoal_planner: AsyncNoGoalPlanner | None = None
        self.system2_client: System2Session | None = None
        self.system2_planner: AsyncSystem2Planner | None = None
        self._intrinsic = np.eye(3, dtype=np.float32)
        self.last_sensor_init_report: dict[str, Any] = {}
        self._legacy_engine = None
        self._legacy_runtime_state = None

    @property
    def navdp_backend_name(self) -> str:
        if self.navdp_client is None:
            return ""
        return str(getattr(self.navdp_client, "backend_name", getattr(self.navdp_client, "backend_impl", "")))

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        self.sensor = self.sensor_factory(
            D455SensorAdapterConfig(
                use_d455=True,
                image_width=int(getattr(self.args, "image_width", 640)),
                image_height=int(getattr(self.args, "image_height", 640)),
                depth_max_m=float(getattr(self.args, "depth_max_m", 5.0)),
                strict_d455=bool(getattr(self.args, "strict_d455", False)),
                force_runtime_mount=bool(getattr(self.args, "force_runtime_camera", False)),
            )
        )
        init_ok, init_msg = self.sensor.initialize(simulation_app, stage)
        self.last_sensor_init_report = {
            "ok": bool(init_ok),
            "message": str(init_msg),
            "capture_report": dict(self.sensor.last_capture_meta),
            "camera_prim_path": str(self.sensor.rgb_prim_path or ""),
            "depth_camera_prim_path": str(self.sensor.depth_prim_path or ""),
            "runtime_mount": bool(self.sensor.runtime_camera_mode),
        }
        print(f"[PLANNING_SESSION] sensor init: ok={init_ok} msg={init_msg}")
        if not init_ok:
            raise RuntimeError(init_msg)
        self.initialize_local(intrinsic=self.sensor.intrinsic)

    def initialize_local(
        self,
        *,
        intrinsic: np.ndarray,
        navdp_client: Any | None = None,
        system2_client: System2Session | None = None,
    ) -> None:
        self._intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
        self.navdp_client = navdp_client or self.navdp_client_factory(self._intrinsic.copy(), self.args)
        self.navdp_client.navigator_reset(self._intrinsic.copy(), batch_size=1)
        self.pointgoal_planner = AsyncPointGoalPlanner(
            client=self.navdp_client,
            use_trajectory_z=bool(getattr(self.args, "use_trajectory_z", False)),
            pointgoal_frame="robot",
        )
        self.nogoal_planner = AsyncNoGoalPlanner(
            client=self.navdp_client,
            use_trajectory_z=bool(getattr(self.args, "use_trajectory_z", False)),
        )
        self.pointgoal_planner.start()
        self.nogoal_planner.start()
        self.system2_client = system2_client or self.system2_client_factory(self.args)
        self.system2_planner = AsyncSystem2Planner(self.system2_client)
        self.system2_planner.start()

    def shutdown(self) -> None:
        for planner in (self.pointgoal_planner, self.nogoal_planner, self.system2_planner):
            if planner is not None:
                planner.stop()

    def ensure_navdp_service_ready(
        self,
        *,
        context: str,
        launcher_processes: dict[str, subprocess.Popen[Any]] | None = None,
    ) -> None:
        try:
            _ensure_remote_service_ready(
                str(getattr(self.args, "server_url", "")),
                timeout_sec=float(getattr(self.args, "timeout_sec", 5.0)),
                service_name="NavDP server",
                context=context,
                launcher_script_name="run_system.ps1",
                launcher_args=("-Component", "nav"),
                launcher_processes=launcher_processes,
                health_path="/healthz",
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} Suggested command: .\\scripts\\run_system.ps1 -Component nav") from exc

    def ensure_system2_service_ready(
        self,
        *,
        context: str,
        launcher_processes: dict[str, subprocess.Popen[Any]] | None = None,
    ) -> None:
        try:
            _ensure_remote_service_ready(
                str(getattr(self.args, "system2_url", "")),
                timeout_sec=float(getattr(self.args, "timeout_sec", 5.0)),
                service_name="System2 server",
                context=context,
                launcher_script_name="run_system.ps1",
                launcher_args=("-Component", "s2"),
                launcher_processes=launcher_processes,
                health_path="/healthz",
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} Suggested command: .\\scripts\\run_system.ps1 -Component s2") from exc

    def capture_observation(self, frame_id: int, *, env=None) -> ExecutionObservation | None:  # noqa: ANN001
        if self.sensor is None:
            raise RuntimeError("PlanningSession.initialize() must be called first.")
        rgb, depth, sensor_meta = self.sensor.capture_rgbd_with_meta(env)
        if rgb is None or depth is None:
            return None
        cam_pos, cam_quat = self.sensor.get_rgb_camera_pose_world()
        if cam_pos is None:
            cam_pos = np.zeros(3, dtype=np.float32)
        if cam_quat is None:
            cam_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return ExecutionObservation(
            frame_id=int(frame_id),
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            sensor_meta=dict(sensor_meta),
            cam_pos=np.asarray(cam_pos, dtype=np.float32),
            cam_quat=np.asarray(cam_quat, dtype=np.float32),
            intrinsic=self._intrinsic.copy(),
        )

    def build_local_observation(
        self,
        *,
        frame_id: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_pose_xyz: tuple[float, float, float] | np.ndarray,
        camera_quat_wxyz: tuple[float, float, float, float] | np.ndarray,
        intrinsic: np.ndarray | None = None,
        sensor_meta: dict[str, Any] | None = None,
        memory_context: MemoryContextBundle | None = None,
    ) -> ExecutionObservation:
        if intrinsic is None:
            intrinsic = self._default_intrinsic(rgb.shape[1], rgb.shape[0])
        return ExecutionObservation(
            frame_id=int(frame_id),
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            sensor_meta=dict(sensor_meta or {}),
            cam_pos=np.asarray(camera_pose_xyz, dtype=np.float32),
            cam_quat=np.asarray(camera_quat_wxyz, dtype=np.float32),
            intrinsic=np.asarray(intrinsic, dtype=np.float32),
            memory_context=memory_context,
        )

    def plan_with_observation(
        self,
        observation: ExecutionObservation,
        *,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> TrajectoryUpdate:
        return self._legacy_planner_engine().plan_with_observation(
            observation,
            action_command=action_command,
            robot_pos_world=robot_pos_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
        )

    def update(
        self,
        frame_id: int,
        *,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
        env=None,  # noqa: ANN001
    ) -> TrajectoryUpdate:
        return self._legacy_planner_engine().update(
            frame_id,
            action_command=action_command,
            robot_pos_world=robot_pos_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
            env=env,
        )

    def _legacy_planner_engine(self):
        if self._legacy_engine is None:
            from server.planner_runtime_engine import PlannerRuntimeEngine
            from server.planner_runtime_state import PlannerRuntimeState

            self._legacy_runtime_state = PlannerRuntimeState(mode=self.mode)
            self._legacy_engine = PlannerRuntimeEngine(self.args, transport=self, state=self._legacy_runtime_state)
        return self._legacy_engine

    @staticmethod
    def _default_intrinsic(width: int, height: int) -> np.ndarray:
        fx = float(width)
        fy = float(height)
        cx = float(width) * 0.5
        cy = float(height) * 0.5
        return np.asarray(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _default_navdp_client_factory(intrinsic: np.ndarray, args: argparse.Namespace) -> InProcessNavDPClient:
        return create_inprocess_navdp_client(
            intrinsic=np.asarray(intrinsic, dtype=np.float32),
            backend=str(getattr(args, "navdp_backend", "heuristic")),
            checkpoint_path=str(getattr(args, "navdp_checkpoint", "")),
            device=str(getattr(args, "navdp_device", "cpu")),
            amp=bool(getattr(args, "navdp_amp", False)),
            amp_dtype=str(getattr(args, "navdp_amp_dtype", "float16")),
            tf32=bool(getattr(args, "navdp_tf32", False)),
            stop_threshold=float(getattr(args, "stop_threshold", -3.0)),
        )

    @staticmethod
    def _default_system2_client_factory(args: argparse.Namespace) -> System2Session:
        language = str(
            getattr(
                args,
                "nav_instruction_language",
                getattr(args, "instruction_language", "auto"),
            )
        )
        return System2Session(
            System2SessionConfig(
                endpoint=str(getattr(args, "system2_url", "http://127.0.0.1:15801")),
                timeout_sec=float(getattr(args, "timeout_sec", 5.0)),
                language=language,
            )
        )
