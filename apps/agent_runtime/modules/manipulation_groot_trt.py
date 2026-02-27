from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import logging
import math
import os
import random
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .direct_locomotion_policy import DirectLocomotionPolicyRuntime
from .g1_action_adapter import G1ActionAdapter
from .hybrid_action_adapter import merge as merge_actions

try:
    from telemetry_runtime import JsonlTelemetryLogger, compute_stats, file_sha256, now_perf
except Exception:  # pragma: no cover - telemetry is optional at runtime
    JsonlTelemetryLogger = None  # type: ignore

    def now_perf() -> float:
        return time.perf_counter()

    def compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"count": 0.0, "mean": None, "std": None, "p50": None, "p95": None, "max": None, "min": None, "rms": None}
        vals = [float(v) for v in values]
        mean = sum(vals) / len(vals)
        rms = math.sqrt(sum(v * v for v in vals) / len(vals))
        return {
            "count": float(len(vals)),
            "mean": float(mean),
            "std": 0.0,
            "p50": float(sorted(vals)[len(vals) // 2]),
            "p95": float(max(vals)),
            "max": float(max(vals)),
            "min": float(min(vals)),
            "rms": float(rms),
        }

    def file_sha256(path: str | Path) -> str:
        return ""

_SONIC_NEUTRAL_JOINTS_29: List[float] = [
    -0.312,
    -0.312,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.669,
    0.669,
    0.2,
    0.2,
    -0.363,
    -0.363,
    0.2,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.6,
    0.6,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


class _Gr00TPolicyServerClient:
    """Minimal ZeroMQ + msgpack client compatible with Isaac-GR00T policy server."""

    def __init__(self, host: str, port: int, timeout_s: float, identity: str) -> None:
        try:
            import msgpack  # type: ignore
            import zmq  # type: ignore
        except Exception as exc:
            raise RuntimeError("Missing dependency for GR00T policy server client: pip install pyzmq msgpack") from exc

        self._zmq = zmq
        self._msgpack = msgpack
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.identity = identity

        self._context = None
        self._socket = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        self._context = self._zmq.Context()
        # Isaac-GR00T PolicyServer uses REQ/REP.
        self._socket = self._context.socket(self._zmq.REQ)
        timeout_ms = int(self.timeout_s * 1000)
        self._socket.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(self._zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(self._zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def get_action(self, observation: Dict[str, Any], task: str, embodiment_tag: str) -> Any:
        if self._socket is None:
            self.connect()
        assert self._socket is not None
        request_payload = {
            "endpoint": "get_action",
            "data": {
                "observation": observation,
                # Current server implementation does not use options.
                "options": {"task": task, "embodiment_tag": embodiment_tag},
            },
        }
        try:
            self._socket.send(
                self._msgpack.packb(
                    request_payload,
                    use_bin_type=True,
                    default=self._encode_custom_classes,
                )
            )
        except self._zmq.Again as exc:
            raise TimeoutError(
                f"Timed out sending request to GR00T policy server {self.host}:{self.port}"
            ) from exc
        try:
            raw = self._socket.recv()
        except self._zmq.Again as exc:
            raise TimeoutError(
                f"Timed out waiting response from GR00T policy server {self.host}:{self.port}"
            ) from exc
        return self._msgpack.unpackb(raw, raw=False, object_hook=self._decode_custom_classes)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    @staticmethod
    def _encode_custom_classes(obj: Any) -> Any:
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        if np is not None and isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj

    @staticmethod
    def _decode_custom_classes(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            try:
                import numpy as np  # type: ignore
            except Exception as exc:
                raise RuntimeError("numpy is required to decode GR00T ndarray payloads.") from exc
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


class _SonicPolicyServerClient:
    """Minimal ZeroMQ + msgpack client for SONIC policy server."""

    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        try:
            import msgpack  # type: ignore
            import zmq  # type: ignore
        except Exception as exc:
            raise RuntimeError("Missing dependency for SONIC policy server client: pip install pyzmq msgpack") from exc

        self._msgpack = msgpack
        self._zmq = zmq
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

        self._context = None
        self._socket = None
        self._last_t_req_send: Optional[float] = None
        self._last_t_resp_recv: Optional[float] = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        self._context = self._zmq.Context()
        self._socket = self._context.socket(self._zmq.REQ)
        timeout_ms = int(self.timeout_s * 1000)
        self._socket.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(self._zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(self._zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def get_action(
        self,
        vx: float,
        vy: float,
        yaw_rate: float,
        style: str,
        joint_pos: List[float],
        joint_vel: List[float],
    ) -> Dict[str, Any]:
        if self._socket is None:
            self.connect()
        assert self._socket is not None

        req = {
            "vx": float(vx),
            "vy": float(vy),
            "yaw_rate": float(yaw_rate),
            "style": str(style),
            "joint_pos": [float(v) for v in joint_pos],
            "joint_vel": [float(v) for v in joint_vel],
        }
        t_req_send = now_perf()
        try:
            self._socket.send(self._msgpack.packb(req, use_bin_type=True))
        except self._zmq.Again as exc:
            raise TimeoutError(
                f"Timed out sending request to SONIC policy server {self.host}:{self.port}"
            ) from exc
        try:
            raw = self._socket.recv()
        except self._zmq.Again as exc:
            raise TimeoutError(
                f"Timed out waiting response from SONIC policy server {self.host}:{self.port}"
            ) from exc
        t_resp_recv = now_perf()
        self._last_t_req_send = t_req_send
        self._last_t_resp_recv = t_resp_recv
        return self._msgpack.unpackb(raw, raw=False, strict_map_key=False)

    def get_last_exchange_timing(self) -> Tuple[Optional[float], Optional[float]]:
        return self._last_t_req_send, self._last_t_resp_recv

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None


class GrootManipulator:
    """GR00T manipulator with policy-server backend, TRT placeholder, and mock fallback."""

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.backend = str(cfg.get("backend", "mock")).strip().lower()
        if bool(cfg.get("mock_mode", False)):
            self.backend = "mock"

        self.fallback_to_mock = bool(cfg.get("fallback_to_mock", True))
        self.warmup_iters = int(cfg.get("warmup_iters", 2))
        self.mock_failure_rate = float(cfg.get("mock_failure_rate", 0.0))
        self.control_dt_s = float(cfg.get("control_dt_s", 0.05))

        self.engine_path = str(cfg.get("engine_path", "models/groot_n1_6_3b_fp8.engine"))
        self.repo_id = str(cfg.get("repo_id", "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"))
        self.local_model_dir = str(cfg.get("local_model_dir", ""))
        self.allow_remote_metadata = bool(cfg.get("allow_remote_metadata", True))
        self.metadata_timeout_s = float(cfg.get("metadata_timeout_s", 10.0))

        policy_cfg = dict(cfg.get("policy_server", {}))
        self.policy_host = str(policy_cfg.get("host", "127.0.0.1"))
        self.policy_port = int(policy_cfg.get("port", 5555))
        self.policy_timeout_s = float(policy_cfg.get("timeout_s", 20.0))
        self.policy_identity = str(policy_cfg.get("identity", "g1-agent-runtime"))
        self.embodiment_tag = str(policy_cfg.get("embodiment_tag", "UNITREE_G1"))
        self.auto_start_policy = bool(policy_cfg.get("auto_start", False))
        self.policy_start_command = str(policy_cfg.get("start_command", ""))
        self.policy_start_cwd = str(policy_cfg.get("start_cwd", ""))
        self.policy_start_timeout_s = float(policy_cfg.get("startup_timeout_s", 30.0))
        self.manage_server_process = bool(policy_cfg.get("manage_server_process", True))
        self.policy_request_retries = max(0, int(policy_cfg.get("request_retries", 2)))
        self.policy_retry_backoff_s = float(policy_cfg.get("retry_backoff_s", 1.0))

        locomotion_backend = str(cfg.get("locomotion_backend", "direct_policy")).strip().lower()
        if locomotion_backend in {"sonic", "sonic_server", "legacy"}:
            locomotion_backend = "sonic_server"
        elif locomotion_backend in {"direct", "direct_policy", "onnx", "policy", "pt"}:
            locomotion_backend = "direct_policy"
        else:
            logging.warning("Unknown locomotion_backend '%s'. Falling back to direct_policy.", locomotion_backend)
            locomotion_backend = "direct_policy"
        self.locomotion_backend_requested = locomotion_backend
        self.locomotion_backend_active = "none"
        self.locomotion_enabled = False
        self._locomotion_abort_token = "LOCOMOTION_ABORT"

        direct_policy_cfg = dict(cfg.get("direct_policy", {}))
        self.direct_policy_enabled = bool(direct_policy_cfg.get("enabled", True))
        self.direct_policy_fallback_to_legacy_sonic = bool(
            direct_policy_cfg.get("fallback_to_legacy_sonic", True)
        )
        self._direct_policy_control_dt_explicit = "control_dt_s" in direct_policy_cfg
        # Keep a single source of truth for control dt between manipulator and policy runtime.
        direct_policy_cfg.setdefault("control_dt_s", float(cfg.get("control_dt_s", 0.05)))
        self.direct_policy_cfg = direct_policy_cfg

        sonic_cfg = dict(cfg.get("sonic_server", {}))
        self.sonic_enabled = bool(sonic_cfg.get("enabled", True))
        self.sonic_host = str(sonic_cfg.get("host", "127.0.0.1"))
        self.sonic_port = int(sonic_cfg.get("port", 5556))
        self.sonic_timeout_s = float(sonic_cfg.get("timeout_s", 5.0))
        self.sonic_style = str(sonic_cfg.get("default_style", "normal"))
        self.sonic_publish_navigate_command = bool(sonic_cfg.get("publish_navigate_command", False))
        self.sonic_control_dt_s = float(sonic_cfg.get("control_dt_s", min(self.control_dt_s, 0.02)))
        if not self._direct_policy_control_dt_explicit:
            self.direct_policy_cfg["control_dt_s"] = float(self.sonic_control_dt_s)
        self.sonic_joint_rate_limit_rad_s = float(sonic_cfg.get("joint_rate_limit_rad_s", 3.0))
        self.sonic_joint_blend_alpha = float(sonic_cfg.get("joint_blend_alpha", 0.6))
        self.sonic_velocity = {
            "vx": float(sonic_cfg.get("default_vx", 0.0)),
            "vy": float(sonic_cfg.get("default_vy", 0.0)),
            "yaw_rate": float(sonic_cfg.get("default_yaw_rate", 0.0)),
            "style": self.sonic_style,
        }
        look_at_cfg = dict(cfg.get("look_at", {}))
        self.look_at_yaw_joint = str(look_at_cfg.get("yaw_joint", "waist_yaw_joint")).strip()
        self.look_at_pitch_joint = str(look_at_cfg.get("pitch_joint", "waist_pitch_joint")).strip()
        yaw_limits = look_at_cfg.get("yaw_limits_deg", [-35.0, 35.0])
        pitch_limits = look_at_cfg.get("pitch_limits_deg", [-20.0, 20.0])
        self.look_at_yaw_min_rad, self.look_at_yaw_max_rad = self._parse_2float_limits_deg(
            yaw_limits,
            default_min=-35.0,
            default_max=35.0,
        )
        self.look_at_pitch_min_rad, self.look_at_pitch_max_rad = self._parse_2float_limits_deg(
            pitch_limits,
            default_min=-20.0,
            default_max=20.0,
        )
        self.look_at_roll_rad = float(look_at_cfg.get("roll_rad", 0.0))
        self._look_at_yaw_rad = 0.0
        self._look_at_pitch_rad = 0.0
        default_joint_map = Path(__file__).resolve().parent / "g1_joint_map.json"
        self.sonic_joint_map_path = Path(str(sonic_cfg.get("joint_map_path", default_joint_map))).resolve()
        self.sonic_joint_names_29 = self._load_sonic_joint_names_29(self.sonic_joint_map_path)
        try:
            self.sonic_joint_map_sha256 = file_sha256(self.sonic_joint_map_path)
        except Exception:
            self.sonic_joint_map_sha256 = ""

        self.image_size = int(cfg.get("image_size", 256))
        telemetry_cfg = dict(cfg.get("telemetry", {}))
        telemetry_enabled = bool(telemetry_cfg.get("enabled", True))
        telemetry_phase = str(telemetry_cfg.get("phase", os.environ.get("AURA_TELEMETRY_PHASE", "standing")))
        telemetry_run_dir = telemetry_cfg.get("run_dir")
        self._telemetry_full_dump_every = max(1, int(telemetry_cfg.get("full_dump_every_steps", 25)))
        if telemetry_enabled and JsonlTelemetryLogger is not None:
            self._telemetry: Optional[JsonlTelemetryLogger] = JsonlTelemetryLogger(
                phase=telemetry_phase,
                component="manipulation_client",
                run_dir=telemetry_run_dir,
                flush_every=max(8, int(telemetry_cfg.get("flush_every", 64))),
                flush_interval_s=float(telemetry_cfg.get("flush_interval_s", 0.5)),
            )
        else:
            self._telemetry = None

        self._video_keys: List[str] = []
        self._state_dims: Dict[str, int] = {}
        self._action_dims: Dict[str, int] = {}
        self._language_keys: List[str] = []
        self._video_horizon = 1
        self._state_horizon = 1
        self._last_state_vectors: Dict[str, List[float]] = {}

        self._policy_client: Optional[_Gr00TPolicyServerClient] = None
        self._sonic_client: Optional[_SonicPolicyServerClient] = None
        self._direct_policy_runtime: Optional[DirectLocomotionPolicyRuntime] = None
        self._policy_process: Optional[subprocess.Popen] = None
        self._mock_runtime = self.backend == "mock"
        self._action_adapter = G1ActionAdapter(dict(cfg.get("action_adapter", {})))
        self._joint_estimate = list(_SONIC_NEUTRAL_JOINTS_29)
        self._joint_velocity_estimate = [0.0] * 29
        self._last_joint_update_ts = time.time()
        self._locomotion_last_publish_ts: Optional[float] = None
        self._locomotion_intervals_s: List[float] = []
        self._locomotion_publish_count = 0
        self._locomotion_pending_sonic_task: Optional[asyncio.Task[List[float]]] = None
        self._locomotion_latest_sonic: Optional[List[float]] = None
        self._locomotion_prev_step_perf: Optional[float] = None
        self._loop_dt_window_s: List[float] = []
        self._sonic_step_idx = 0
        self._logged_body29_names = False
        self._last_sonic_call_meta: Dict[str, Any] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    async def warmup(self) -> None:
        if self.backend == "gr00t_policy_server":
            self._load_modality_metadata()
            if self.auto_start_policy:
                self._ensure_policy_server_running()
            self._init_policy_client()
            logging.info(
                "GR00T policy backend ready: repo=%s embodiment=%s host=%s port=%s",
                self.repo_id,
                self.embodiment_tag,
                self.policy_host,
                self.policy_port,
            )
        elif self.backend == "trt_engine":
            if not Path(self.engine_path).exists():
                logging.warning("GR00T engine not found (%s).", self.engine_path)
                if self.fallback_to_mock:
                    self._mock_runtime = True
                    logging.warning("Fallback to mock manipulation mode.")
            else:
                logging.info("Loading GR00T TensorRT engine: %s", self.engine_path)
                logging.warning("TODO: Implement in-process TensorRT GR00T inference.")
        else:
            self._mock_runtime = True
            logging.info("GR00T mock mode enabled.")

        self.locomotion_enabled = False
        self.locomotion_backend_active = "none"
        direct_runtime_meta: Dict[str, Any] = {}

        if self.locomotion_backend_requested == "direct_policy" and self.direct_policy_enabled:
            try:
                self._init_direct_policy_runtime()
                assert self._direct_policy_runtime is not None
                self.locomotion_enabled = True
                self.locomotion_backend_active = "direct_policy"
                direct_runtime_meta = {
                    "direct_policy_path": str(self._direct_policy_runtime.policy_path),
                    "direct_policy_backend": str(self._direct_policy_runtime.backend),
                    "direct_policy_obs_dim": int(self._direct_policy_runtime.obs_dim),
                    "direct_policy_action_dim": int(self._direct_policy_runtime.raw_action_dim),
                }
                logging.info(
                    "Direct locomotion backend ready: path=%s backend=%s obs_dim=%s action_dim=%s",
                    direct_runtime_meta["direct_policy_path"],
                    direct_runtime_meta["direct_policy_backend"],
                    direct_runtime_meta["direct_policy_obs_dim"],
                    direct_runtime_meta["direct_policy_action_dim"],
                )
            except Exception as exc:
                logging.warning("Direct locomotion backend unavailable: %s", exc)
                if self.sonic_enabled and self.direct_policy_fallback_to_legacy_sonic:
                    try:
                        self._init_sonic_client()
                        self.locomotion_enabled = True
                        self.locomotion_backend_active = "sonic_server"
                        logging.warning(
                            "Falling back to legacy SONIC backend: host=%s port=%s style=%s",
                            self.sonic_host,
                            self.sonic_port,
                            self.sonic_velocity["style"],
                        )
                    except Exception as sonic_exc:
                        self.sonic_enabled = False
                        logging.warning("SONIC backend unavailable after direct-policy fallback: %s", sonic_exc)
        elif self.sonic_enabled:
            try:
                self._init_sonic_client()
                self.locomotion_enabled = True
                self.locomotion_backend_active = "sonic_server"
                logging.info(
                    "SONIC backend ready: host=%s port=%s style=%s",
                    self.sonic_host,
                    self.sonic_port,
                    self.sonic_velocity["style"],
                )
            except Exception as exc:
                self.sonic_enabled = False
                logging.warning("SONIC backend unavailable. Falling back to navigate-only locomotion: %s", exc)
        self._telemetry_log(
            {
                "event": "warmup_config",
                "joint_map_path": str(self.sonic_joint_map_path),
                "joint_map_sha256": self.sonic_joint_map_sha256,
                "applied_dof_count": len(self.sonic_joint_names_29),
                "applied_body29_dof_names": list(self.sonic_joint_names_29),
                "locomotion_backend_requested": self.locomotion_backend_requested,
                "locomotion_backend_active": self.locomotion_backend_active,
                "locomotion_enabled": bool(self.locomotion_enabled),
                "sonic_enabled": bool(self.sonic_enabled),
                "sonic_host": self.sonic_host,
                "sonic_port": self.sonic_port,
                "sonic_control_dt_s": self.sonic_control_dt_s,
                **direct_runtime_meta,
            }
        )

        for i in range(self.warmup_iters):
            await asyncio.sleep(0.20)
            logging.info("GR00T warmup iteration %s/%s", i + 1, self.warmup_iters)

    def update_state_vectors(self, state_vectors: Dict[str, List[float]]) -> None:
        self._last_state_vectors = dict(state_vectors)

    def set_sonic_velocity(self, command: Any) -> None:
        if isinstance(command, dict):
            vx = float(command.get("vx", self.sonic_velocity["vx"]))
            vy = float(command.get("vy", self.sonic_velocity["vy"]))
            yaw_rate = float(command.get("yaw_rate", self.sonic_velocity["yaw_rate"]))
            style = str(command.get("style", self.sonic_velocity["style"]))
        else:
            vx = float(getattr(command, "vx", self.sonic_velocity["vx"]))
            vy = float(getattr(command, "vy", self.sonic_velocity["vy"]))
            yaw_rate = float(getattr(command, "yaw_rate", self.sonic_velocity["yaw_rate"]))
            style = str(getattr(command, "style", self.sonic_velocity["style"]))
        self.sonic_velocity = {"vx": vx, "vy": vy, "yaw_rate": yaw_rate, "style": style}

    @staticmethod
    def _parse_2float_limits_deg(raw: Any, default_min: float, default_max: float) -> Tuple[float, float]:
        lo = float(default_min)
        hi = float(default_max)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            try:
                lo = float(raw[0])
                hi = float(raw[1])
            except Exception:
                lo = float(default_min)
                hi = float(default_max)
        if lo > hi:
            lo, hi = hi, lo
        return math.radians(lo), math.radians(hi)

    def get_camera_aim(self) -> Tuple[float, float]:
        feedback = self._action_adapter.get_last_joint_feedback()
        if feedback is None:
            return float(self._look_at_yaw_rad), float(self._look_at_pitch_rad)

        names = [str(v) for v in feedback.get("name", [])]
        pos = [float(v) for v in feedback.get("position", [])]
        if not names or not pos:
            return float(self._look_at_yaw_rad), float(self._look_at_pitch_rad)

        name_to_idx = {name: idx for idx, name in enumerate(names)}
        yaw_idx = name_to_idx.get(self.look_at_yaw_joint)
        pitch_idx = name_to_idx.get(self.look_at_pitch_joint)
        if yaw_idx is not None and yaw_idx < len(pos):
            self._look_at_yaw_rad = float(pos[yaw_idx])
        if pitch_idx is not None and pitch_idx < len(pos):
            self._look_at_pitch_rad = float(pos[pitch_idx])
        return float(self._look_at_yaw_rad), float(self._look_at_pitch_rad)

    def command_camera_aim(self, yaw_rad: float, pitch_rad: float, source: str = "look_at") -> Tuple[float, float]:
        yaw = float(max(self.look_at_yaw_min_rad, min(self.look_at_yaw_max_rad, float(yaw_rad))))
        pitch = float(max(self.look_at_pitch_min_rad, min(self.look_at_pitch_max_rad, float(pitch_rad))))
        self._look_at_yaw_rad = yaw
        self._look_at_pitch_rad = pitch
        self._action_adapter.apply_action(
            {"waist": [float(yaw), float(self.look_at_roll_rad), float(pitch)]},
            source=source,
        )
        return yaw, pitch

    @staticmethod
    def _load_sonic_joint_names_29(map_path: Path) -> List[str]:
        payload = json.loads(map_path.read_text(encoding="utf-8"))
        joints = payload.get("joints", [])
        if not isinstance(joints, list):
            raise RuntimeError(f"Invalid joint map format: {map_path}")
        ordered = sorted(joints, key=lambda item: int(item.get("sonic_idx", 0)))
        names = [str(item.get("name", "")).strip() for item in ordered if str(item.get("name", "")).strip()]
        if len(names) < 29:
            raise RuntimeError(f"Joint map contains {len(names)} joints, expected >=29: {map_path}")
        return names[:29]

    def set_telemetry_phase(self, phase: str) -> None:
        if self._telemetry is not None:
            self._telemetry.switch_phase(str(phase))
        self._sonic_step_idx = 0
        self._loop_dt_window_s.clear()
        self._locomotion_intervals_s.clear()
        self._locomotion_last_publish_ts = None

    def get_telemetry_run_dir(self) -> Optional[Path]:
        if self._telemetry is None:
            return None
        return Path(self._telemetry.run_dir)

    def get_runtime_metadata(self) -> Dict[str, Any]:
        meta = {
            "joint_map_path": str(self.sonic_joint_map_path),
            "joint_map_sha256": self.sonic_joint_map_sha256,
            "sonic_joint_names_29": list(self.sonic_joint_names_29),
            "sonic_control_dt_s": float(self.sonic_control_dt_s),
            "locomotion_backend_requested": str(self.locomotion_backend_requested),
            "locomotion_backend_active": str(self.locomotion_backend_active),
            "locomotion_enabled": bool(self.locomotion_enabled),
        }
        if self._direct_policy_runtime is not None:
            meta["direct_policy_path"] = str(self._direct_policy_runtime.policy_path)
            meta["direct_policy_backend"] = str(self._direct_policy_runtime.backend)
            meta["direct_policy_obs_dim"] = int(self._direct_policy_runtime.obs_dim)
            meta["direct_policy_action_dim"] = int(self._direct_policy_runtime.raw_action_dim)
        return meta

    def _telemetry_log(self, record: Dict[str, Any]) -> None:
        if self._telemetry is None:
            return
        try:
            self._telemetry.log(record)
        except Exception as exc:
            logging.debug("Telemetry logging failed: %s", exc)

    @staticmethod
    def _max_abs(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(max(abs(float(v)) for v in values))

    @staticmethod
    def _flatten_joint_payload_29(payload: Any) -> List[float]:
        if payload is None:
            return []
        arr = payload
        if isinstance(arr, (list, tuple)) and len(arr) == 1 and isinstance(arr[0], (list, tuple)):
            arr = arr[0]
        if isinstance(arr, (list, tuple)):
            values = [float(v) for v in list(arr)]
            return values[:29]
        return []

    @staticmethod
    def _is_all_zero(values: List[float], eps: float = 1e-8) -> bool:
        if not values:
            return True
        return all(abs(float(v)) <= eps for v in values)

    def _is_locomotion_command_active(self) -> bool:
        return (
            abs(float(self.sonic_velocity["vx"]))
            + abs(float(self.sonic_velocity["vy"]))
            + abs(float(self.sonic_velocity["yaw_rate"]))
        ) > 1e-3

    def _build_sonic_request_from_live_feedback(self) -> Tuple[List[float], List[float], Dict[str, Any]]:
        feedback = self._action_adapter.get_last_joint_feedback()
        if feedback is None:
            pos, vel = self._estimate_joint_state()
            return pos, vel, {"feedback_source": "estimate", "feedback_missing": True}

        names_raw = [str(v) for v in feedback.get("name", [])]
        pos_raw = [float(v) for v in feedback.get("position", [])]
        vel_raw = [float(v) for v in feedback.get("velocity", [])]
        if not names_raw or not pos_raw:
            pos, vel = self._estimate_joint_state()
            return pos, vel, {"feedback_source": "estimate", "feedback_missing": True}

        if len(vel_raw) < len(pos_raw):
            vel_raw.extend([0.0] * (len(pos_raw) - len(vel_raw)))
        name_to_idx: Dict[str, int] = {}
        for idx, name in enumerate(names_raw):
            if name not in name_to_idx:
                name_to_idx[name] = idx

        pos_29: List[float] = []
        vel_29: List[float] = []
        missing: List[str] = []
        for joint_name in self.sonic_joint_names_29:
            idx = name_to_idx.get(joint_name)
            if idx is None or idx >= len(pos_raw):
                missing.append(joint_name)
                pos_29.append(0.0)
                vel_29.append(0.0)
                continue
            pos_29.append(float(pos_raw[idx]))
            vel_29.append(float(vel_raw[idx]))

        if missing:
            pos_est, vel_est = self._estimate_joint_state()
            for i, joint_name in enumerate(self.sonic_joint_names_29):
                if joint_name in missing and i < len(pos_est):
                    pos_29[i] = float(pos_est[i])
                    vel_29[i] = float(vel_est[i]) if i < len(vel_est) else 0.0
            return (
                pos_29,
                vel_29,
                {
                    "feedback_source": "joint_states_partial",
                    "feedback_missing": True,
                    "missing_names": missing,
                },
            )

        return (
            pos_29[:29],
            vel_29[:29],
            {
                "feedback_source": "joint_states",
                "feedback_missing": False,
                "joint_state_count": len(names_raw),
                "feedback_timestamp": float(feedback.get("timestamp", 0.0)),
            },
        )

    def _dump_recent_abort_window(self, stem: str, ref_perf: float) -> Optional[Path]:
        if self._telemetry is None:
            return None
        try:
            return self._telemetry.dump_recent_window(
                filename=f"{stem}_last1s.jsonl",
                within_s=1.0,
                ref_time=ref_perf,
            )
        except Exception as exc:
            logging.warning("Failed to dump recent telemetry window: %s", exc)
            return None

    async def pick(
        self,
        target: str,
        image: Optional[Any] = None,
        target_box: Optional[Any] = None,
        instruction: str = "",
        pause_event: Optional[asyncio.Event] = None,
    ) -> bool:
        text = instruction or f"Pick {target} and hold safely."
        return await self._run_skill("pick", target, text, image, target_box, pause_event)

    async def inspect(
        self,
        target: str,
        image: Optional[Any] = None,
        target_box: Optional[Any] = None,
        instruction: str = "",
        pause_event: Optional[asyncio.Event] = None,
    ) -> bool:
        text = instruction or f"Inspect {target} and report condition."
        return await self._run_skill("inspect", target, text, image, target_box, pause_event)

    async def execute_instruction(
        self,
        instruction: str,
        image: Optional[Any] = None,
        target_box: Optional[Any] = None,
        pause_event: Optional[asyncio.Event] = None,
    ) -> bool:
        text = instruction.strip()
        if not text:
            return False
        return await self._run_skill(
            action="instruction",
            target="",
            instruction=text,
            image=image,
            target_box=target_box,
            pause_event=pause_event,
        )

    async def execute_locomotion(
        self,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
        duration_s: float = 2.0,
        pause_event: Optional[asyncio.Event] = None,
        source: str = "manual_locomotion",
    ) -> bool:
        """Publishes policy-driven whole-body commands for locomotion."""
        dt = max(0.01, float(self.sonic_control_dt_s if self.locomotion_enabled else self.control_dt_s))
        total_s = max(dt, float(duration_s))
        steps = max(1, int(total_s / dt))
        self.set_sonic_velocity(
            {"vx": float(linear_x), "vy": float(linear_y), "yaw_rate": float(angular_z), "style": self.sonic_style}
        )
        cmd_nav = [float(linear_x), float(linear_y), float(angular_z)]
        pending_sonic = self._locomotion_pending_sonic_task
        latest_sonic = self._locomotion_latest_sonic
        if pending_sonic is not None:
            if pending_sonic.cancelled():
                pending_sonic = None
            elif pending_sonic.done():
                try:
                    latest_sonic = pending_sonic.result()
                except Exception as exc:
                    if "SONIC_ABORT" in str(exc) or self._locomotion_abort_token in str(exc):
                        raise
                    logging.warning("Locomotion policy request failed. Reusing previous target: %s", exc)
                finally:
                    pending_sonic = None
        next_tick = time.perf_counter()
        self._locomotion_prev_step_perf = None

        for _ in range(steps):
            t_start = time.perf_counter()
            if self._locomotion_prev_step_perf is None:
                loop_dt = dt
            else:
                loop_dt = max(0.0, t_start - self._locomotion_prev_step_perf)
            self._locomotion_prev_step_perf = t_start
            self._loop_dt_window_s.append(loop_dt)
            if len(self._loop_dt_window_s) > 400:
                self._loop_dt_window_s = self._loop_dt_window_s[-400:]
            if pause_event is not None and not pause_event.is_set():
                logging.info("Locomotion paused due to SLAM exploration mode.")
                await pause_event.wait()

            t_apply = None
            if self.locomotion_enabled:
                if pending_sonic is None:
                    pending_sonic = asyncio.create_task(asyncio.to_thread(self._request_locomotion_target))

                if pending_sonic.done():
                    try:
                        latest_sonic = pending_sonic.result()
                    except Exception as exc:
                        if "SONIC_ABORT" in str(exc) or self._locomotion_abort_token in str(exc):
                            raise
                        logging.warning("Locomotion policy request failed. Reusing previous target: %s", exc)
                    finally:
                        pending_sonic = asyncio.create_task(asyncio.to_thread(self._request_locomotion_target))

                if latest_sonic is not None:
                    merged = merge_actions(
                        groot_actions={},
                        sonic_actions=latest_sonic,
                        mode="locomotion",
                    )
                    stabilized = self._stabilize_sonic_targets(merged.tolist(), dt)
                else:
                    stabilized = list(self._joint_estimate)

                self._cache_joint_state(stabilized)
                payload: Dict[str, Any] = {"joint_command": stabilized}
                if self.sonic_publish_navigate_command:
                    payload["navigate_command"] = cmd_nav
                self._action_adapter.apply_action(payload, source=source)
                t_apply = now_perf()
                self._record_locomotion_publish(t_apply)
            else:
                self._action_adapter.apply_action({"navigate_command": cmd_nav}, source=source)
                t_apply = now_perf()
                self._record_locomotion_publish(t_apply)

            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            overrun_ms = 0.0
            if sleep_time > 0.0:
                # Coarse sleep first, then fine-grained yield loop for tighter cadence.
                if sleep_time > 0.002:
                    await asyncio.sleep(sleep_time - 0.001)
                while True:
                    remaining = next_tick - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    await asyncio.sleep(0)
            else:
                elapsed = time.perf_counter() - t_start
                overrun_ms = elapsed * 1000.0 - (dt * 1000.0)
                if overrun_ms > 2.0:
                    logging.warning(
                        "Locomotion control loop overrun: %.1f ms (budget %.1f ms)",
                        elapsed * 1000.0,
                        dt * 1000.0,
                    )
                else:
                    logging.debug(
                        "Locomotion loop jitter: %.2f ms over budget %.1f ms",
                        overrun_ms,
                        dt * 1000.0,
                    )
                next_tick = time.perf_counter()
            publish_hz_window = None
            if self._locomotion_intervals_s:
                window = self._locomotion_intervals_s[-50:]
                avg_interval = sum(window) / len(window)
                if avg_interval > 0.0:
                    publish_hz_window = 1.0 / avg_interval
            step_meta = dict(self._last_sonic_call_meta) if self._last_sonic_call_meta else {}
            record: Dict[str, Any] = {
                "event": "locomotion_apply",
                "step_idx": int(self._locomotion_publish_count),
                "vx": float(self.sonic_velocity["vx"]),
                "vy": float(self.sonic_velocity["vy"]),
                "yaw_rate": float(self.sonic_velocity["yaw_rate"]),
                "style": str(self.sonic_velocity["style"]),
                "loop_dt": float(loop_dt),
                "loop_overrun_ms": float(overrun_ms),
                "publish_hz_window": publish_hz_window,
            }
            if t_apply is not None:
                record["t_apply"] = float(t_apply)
            if "server_latency_ms" in step_meta:
                record["server_latency_ms"] = step_meta["server_latency_ms"]
            self._telemetry_log(record)

        self._locomotion_pending_sonic_task = pending_sonic
        self._locomotion_latest_sonic = latest_sonic

        if self.sonic_publish_navigate_command or not self.locomotion_enabled:
            self._action_adapter.apply_action(
                {"navigate_command": [0.0, 0.0, 0.0]},
                source=f"{source}:stop",
            )
        return True

    def _record_locomotion_publish(self, publish_ts: float) -> None:
        ts = float(publish_ts)
        if self._locomotion_last_publish_ts is not None:
            interval = ts - self._locomotion_last_publish_ts
            if interval > 0.0:
                self._locomotion_intervals_s.append(interval)
                if len(self._locomotion_intervals_s) > 400:
                    self._locomotion_intervals_s = self._locomotion_intervals_s[-400:]
        self._locomotion_last_publish_ts = ts
        self._locomotion_publish_count += 1

        if self._locomotion_publish_count % 100 == 0 and len(self._locomotion_intervals_s) >= 100:
            avg_interval = sum(self._locomotion_intervals_s[-100:]) / 100.0
            if avg_interval > 0.0:
                avg_hz = 1.0 / avg_interval
                logging.info("[FREQ] Average publish rate over last 100 steps: %.1f Hz", avg_hz)

    async def stop(self) -> None:
        if self._locomotion_pending_sonic_task is not None:
            pending = self._locomotion_pending_sonic_task
            try:
                if not pending.done():
                    await asyncio.wait_for(
                        asyncio.shield(pending),
                        timeout=max(0.5, float(self.sonic_timeout_s) + 0.5),
                    )
                if pending.done() and not pending.cancelled():
                    try:
                        latest = pending.result()
                        self._locomotion_latest_sonic = [float(v) for v in latest]
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                logging.warning("Timeout waiting for pending locomotion request during stop; cancelling it.")
                pending.cancel()
            except Exception as exc:
                logging.debug("Pending SONIC task cleanup failed during stop: %s", exc)
            finally:
                self._locomotion_pending_sonic_task = None

        if self._policy_client is not None:
            self._policy_client.close()
            self._policy_client = None
        if self._sonic_client is not None:
            self._sonic_client.close()
            self._sonic_client = None
        if self._direct_policy_runtime is not None:
            self._direct_policy_runtime.close()
            self._direct_policy_runtime = None
        if self._policy_process is not None and self.manage_server_process:
            try:
                self._policy_process.terminate()
            except Exception:
                pass
            self._policy_process = None
        self._executor.shutdown(wait=True)
        self._action_adapter.close()
        if self._telemetry is not None:
            try:
                self._telemetry.flush()
                self._telemetry.close()
            except Exception:
                pass

    async def _run_skill(
        self,
        action: str,
        target: str,
        instruction: str,
        image: Optional[Any],
        target_box: Optional[Any],
        pause_event: Optional[asyncio.Event],
    ) -> bool:
        if self.backend == "gr00t_policy_server" and not self._mock_runtime:
            try:
                ok = await self._run_policy_server_action(
                    action=action,
                    target=target,
                    instruction=instruction,
                    image=image,
                    target_box=target_box,
                    pause_event=pause_event,
                )
                if ok:
                    return True
            except Exception as exc:
                logging.warning("GR00T policy action failed: %s", exc)

            if not self.fallback_to_mock:
                return False
            logging.warning("Switching manipulation to mock fallback.")
            self._mock_runtime = True

        if self.backend == "trt_engine" and not self._mock_runtime:
            logging.warning("TODO: Execute GR00T TensorRT inference and map output to G1 control adapter.")
            await asyncio.sleep(0.2)
            return False

        return await self._run_mock(action, target, instruction, pause_event)

    async def _run_policy_server_action(
        self,
        action: str,
        target: str,
        instruction: str,
        image: Optional[Any],
        target_box: Optional[Any],
        pause_event: Optional[asyncio.Event],
    ) -> bool:
        if not self._is_tcp_open(self.policy_host, self.policy_port, timeout_s=0.2):
            logging.warning(
                "GR00T policy server not reachable at %s:%s.",
                self.policy_host,
                self.policy_port,
            )
            return False

        if self._policy_client is None:
            self._init_policy_client()
        if self._policy_client is None:
            raise RuntimeError("Policy client is not initialized.")

        task = self._compose_task(action, target, instruction)
        obs = self._build_observation(image=image, instruction=task)
        attempts = self.policy_request_retries + 1
        response: Optional[Dict[str, Any]] = None
        last_exc: Optional[Exception] = None
        seed_sonic_action = None
        for attempt in range(1, attempts + 1):
            try:
                if self._policy_client is None:
                    self._init_policy_client()
                assert self._policy_client is not None
                loop = asyncio.get_running_loop()
                groot_future = loop.run_in_executor(
                    self._executor,
                    self._policy_client.get_action,
                    obs,
                    task,
                    self.embodiment_tag,
                )
                sonic_future = None
                if self.locomotion_enabled:
                    sonic_future = loop.run_in_executor(self._executor, self._request_locomotion_target)
                response = await groot_future
                if sonic_future is not None:
                    try:
                        seed_sonic_action = await sonic_future
                    except Exception as sonic_exc:
                        if "SONIC_ABORT" in str(sonic_exc) or self._locomotion_abort_token in str(sonic_exc):
                            raise
                        logging.debug("Initial locomotion prefetch failed: %s", sonic_exc)
                break
            except Exception as exc:
                last_exc = exc
                logging.warning(
                    "GR00T policy request failed (%s/%s, timeout=%.1fs): %s",
                    attempt,
                    attempts,
                    self.policy_timeout_s,
                    exc,
                )
                self._reset_policy_client()
                if attempt < attempts:
                    await asyncio.sleep(max(0.0, self.policy_retry_backoff_s))
        if response is None:
            raise RuntimeError(
                f"No GR00T policy response after {attempts} attempts: {last_exc}"
            )

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")

        actions_payload: Optional[Dict[str, Any]] = None
        if isinstance(response, dict) and "actions" in response:
            # Legacy/custom response shape.
            actions_payload = response.get("actions")
        elif isinstance(response, (list, tuple)) and response:
            # Isaac-GR00T PolicyServer returns (action, info), msgpack-unpacked as list.
            first = response[0]
            if isinstance(first, dict):
                actions_payload = first
        elif isinstance(response, dict):
            # Some deployments may return the action dict directly.
            actions_payload = response

        if not actions_payload:
            logging.warning("GR00T response did not include action payload: %s", response)
            return False

        action_steps = self._extract_action_steps(actions_payload)
        if not action_steps:
            logging.warning("GR00T action payload was empty.")
            return False

        logging.info(
            "GR00T action received: steps=%s keys=%s",
            len(action_steps),
            sorted(action_steps[0].keys()),
        )

        for idx, step in enumerate(action_steps):
            if pause_event is not None and not pause_event.is_set():
                logging.info("Manipulation paused due to SLAM exploration mode.")
                await pause_event.wait()

            sonic_action = None
            if self.locomotion_enabled:
                if idx == 0 and seed_sonic_action is not None:
                    sonic_action = seed_sonic_action
                else:
                    try:
                        sonic_action = await asyncio.to_thread(self._request_locomotion_target)
                    except Exception as sonic_exc:
                        if "SONIC_ABORT" in str(sonic_exc) or self._locomotion_abort_token in str(sonic_exc):
                            raise
                        logging.debug("Locomotion request failed during manipulation step: %s", sonic_exc)

            if sonic_action is None:
                sonic_action = self._joint_estimate

            merged = merge_actions(
                groot_actions=step,
                sonic_actions=sonic_action,
                mode="manipulation",
            )
            self._cache_joint_state(merged.tolist())
            payload = dict(step)
            payload["joint_command"] = merged.tolist()
            if self.sonic_publish_navigate_command:
                payload["navigate_command"] = [
                    float(self.sonic_velocity["vx"]),
                    float(self.sonic_velocity["vy"]),
                    float(self.sonic_velocity["yaw_rate"]),
                ]
            self._action_adapter.apply_action(
                payload,
                source=f"gr00t+{self.locomotion_backend_active}:{self.embodiment_tag}",
            )
            await asyncio.sleep(self.control_dt_s)

        return True

    async def _run_mock(
        self, action: str, target: str, instruction: str, pause_event: Optional[asyncio.Event]
    ) -> bool:
        logging.info("GR00T mock action=%s target=%s instruction=%s", action, target, instruction)
        for _ in range(8):
            if pause_event is not None and not pause_event.is_set():
                logging.info("Manipulation paused due to SLAM exploration mode.")
                await pause_event.wait()
            await asyncio.sleep(0.15)
        if random.random() < self.mock_failure_rate:
            return False
        return True

    def _compose_task(self, action: str, target: str, instruction: str) -> str:
        if instruction.strip():
            return instruction.strip()
        if action == "pick":
            return f"Pick up the {target} and place it on the plate."
        if action == "inspect":
            return f"Inspect the {target} while maintaining safe posture."
        return f"{action} {target}"

    def _init_policy_client(self) -> None:
        self._policy_client = _Gr00TPolicyServerClient(
            host=self.policy_host,
            port=self.policy_port,
            timeout_s=self.policy_timeout_s,
            identity=self.policy_identity,
        )
        self._policy_client.connect()

    def _reset_policy_client(self) -> None:
        if self._policy_client is not None:
            try:
                self._policy_client.close()
            except Exception:
                pass
        self._policy_client = None

    def _init_sonic_client(self) -> None:
        self._sonic_client = _SonicPolicyServerClient(
            host=self.sonic_host,
            port=self.sonic_port,
            timeout_s=self.sonic_timeout_s,
        )
        self._sonic_client.connect()

    def _init_direct_policy_runtime(self) -> None:
        self._direct_policy_runtime = DirectLocomotionPolicyRuntime(self.direct_policy_cfg)

    def _request_locomotion_target(self) -> List[float]:
        if self.locomotion_backend_active == "direct_policy":
            return self._request_direct_policy()
        return self._request_sonic()

    def _cache_joint_state(self, joint_cmd: List[float]) -> None:
        now = time.time()
        dt = max(1e-4, now - self._last_joint_update_ts)
        if len(self._joint_estimate) != len(joint_cmd):
            self._joint_estimate = [0.0] * len(joint_cmd)
            self._joint_velocity_estimate = [0.0] * len(joint_cmd)
        self._joint_velocity_estimate = [
            float(joint_cmd[i] - self._joint_estimate[i]) / dt for i in range(len(joint_cmd))
        ]
        self._joint_estimate = [float(v) for v in joint_cmd]
        self._last_joint_update_ts = now

    def _estimate_joint_state(self) -> Tuple[List[float], List[float]]:
        latest = self._action_adapter.get_last_joint_command()
        if latest is not None and len(latest) == len(self._joint_estimate):
            self._cache_joint_state([float(v) for v in latest])
        return list(self._joint_estimate), list(self._joint_velocity_estimate)

    def _request_direct_policy(self) -> List[float]:
        if self._direct_policy_runtime is None:
            self._init_direct_policy_runtime()
        assert self._direct_policy_runtime is not None

        step_idx = int(self._sonic_step_idx)
        self._sonic_step_idx += 1
        vx = float(self.sonic_velocity["vx"])
        vy = float(self.sonic_velocity["vy"])
        yaw_rate = float(self.sonic_velocity["yaw_rate"])
        style = str(self.sonic_velocity["style"])

        joint_pos, joint_vel, feedback_meta = self._build_sonic_request_from_live_feedback()
        if len(joint_pos) < 29:
            joint_pos.extend([0.0] * (29 - len(joint_pos)))
        if len(joint_vel) < 29:
            joint_vel.extend([0.0] * (29 - len(joint_vel)))
        joint_pos = [float(v) for v in joint_pos[:29]]
        joint_vel = [float(v) for v in joint_vel[:29]]
        req_is_all_zero_pos = self._is_all_zero(joint_pos, eps=1e-8)
        req_is_all_zero_vel = self._is_all_zero(joint_vel, eps=1e-8)
        req_max_abs_pos = self._max_abs(joint_pos)
        req_max_abs_vel = self._max_abs(joint_vel)

        if self._is_locomotion_command_active() and req_is_all_zero_pos:
            msg = (
                f"{self._locomotion_abort_token}: locomotion command requested while req_joint_pos_29 is all zeros. "
                "Check live Isaac feedback mapping and joint-state stream."
            )
            self._telemetry_log(
                {
                    "event": "direct_policy_request_rejected",
                    "step_idx": step_idx,
                    "vx": vx,
                    "vy": vy,
                    "yaw_rate": yaw_rate,
                    "style": style,
                    "req_is_all_zero_pos": True,
                    "req_is_all_zero_vel": req_is_all_zero_vel,
                    "req_max_abs_pos": req_max_abs_pos,
                    "req_max_abs_vel": req_max_abs_vel,
                    "feedback_source": feedback_meta.get("feedback_source"),
                    "error": msg,
                }
            )
            raise RuntimeError(msg)

        base_pose = self._action_adapter.get_last_base_pose() or {}
        out = self._direct_policy_runtime.infer(
            vx=vx,
            vy=vy,
            yaw_rate=yaw_rate,
            joint_pos_29=joint_pos,
            joint_vel_29=joint_vel,
            base_pose=base_pose,
        )
        target_joint_pos_29 = [float(v) for v in list(out.get("joint_actions_29", []))[:29]]
        if len(target_joint_pos_29) < 29:
            raise RuntimeError("Direct locomotion policy did not return 29 joint targets.")
        raw_action_29 = [float(v) for v in list(out.get("raw_actions_29", []))[:29]]
        raw_action_dim = int(out.get("raw_action_dim", len(raw_action_29)))
        obs_dim = int(out.get("obs_dim", 0))
        obs_used_dim = int(out.get("obs_used_dim", 0))
        infer_latency_ms = float(out.get("infer_latency_ms", 0.0))

        raw_max_abs_action = self._max_abs(raw_action_29) if raw_action_29 else None
        target_max_abs = self._max_abs(target_joint_pos_29)
        resp_max_abs_action = float(raw_max_abs_action if raw_max_abs_action is not None else target_max_abs)

        stabilized = self._stabilize_sonic_targets(target_joint_pos_29, self.sonic_control_dt_s)
        delta_before = self._max_abs(
            [target_joint_pos_29[i] - self._joint_estimate[i] for i in range(len(self._joint_estimate))]
        )
        delta_after = self._max_abs([stabilized[i] - self._joint_estimate[i] for i in range(len(self._joint_estimate))])
        tracking_err = [stabilized[i] - joint_pos[i] for i in range(min(len(stabilized), len(joint_pos)))]
        tracking_stats = compute_stats(tracking_err)
        tracking_rms = tracking_stats.get("rms")
        base_roll = float(base_pose.get("roll", 0.0))
        base_pitch = float(base_pose.get("pitch", 0.0))
        base_yaw = float(base_pose.get("yaw", 0.0))
        base_height = float(base_pose.get("z", 0.0))
        fall_flag = bool(
            (abs(base_roll) > 0.9)
            or (abs(base_pitch) > 0.9)
            or (base_height > 0.0 and base_height < 0.22)
        )
        slip_flag = bool(self._is_locomotion_command_active() and req_max_abs_vel < 1e-3 and req_max_abs_pos > 1e-4)

        log_full = (step_idx % self._telemetry_full_dump_every) == 0
        record: Dict[str, Any] = {
            "event": "direct_policy_exchange",
            "step_idx": step_idx,
            "vx": vx,
            "vy": vy,
            "yaw_rate": yaw_rate,
            "style": style,
            "req_is_all_zero_pos": req_is_all_zero_pos,
            "req_is_all_zero_vel": req_is_all_zero_vel,
            "req_max_abs_pos": req_max_abs_pos,
            "req_max_abs_vel": req_max_abs_vel,
            "resp_max_abs_action": resp_max_abs_action,
            "raw_max_abs_action": raw_max_abs_action,
            "target_max_abs": target_max_abs,
            "delta_max_abs_before_clip": delta_before,
            "delta_max_abs_after_clip": delta_after,
            "tracking_err_max_abs": self._max_abs(tracking_err),
            "tracking_err_rms": float(tracking_rms) if tracking_rms is not None else None,
            "base_roll": base_roll,
            "base_pitch": base_pitch,
            "base_yaw": base_yaw,
            "base_height": base_height,
            "fall_flag": fall_flag,
            "slip_flag": slip_flag,
            "server_latency_ms": infer_latency_ms,
            "policy_infer_latency_ms": infer_latency_ms,
            "policy_obs_dim": obs_dim,
            "policy_obs_used_dim": obs_used_dim,
            "policy_raw_action_dim": raw_action_dim,
            "applied_dof_count": len(self.sonic_joint_names_29),
            "applied_target_len": len(stabilized),
            "feedback_source": feedback_meta.get("feedback_source"),
            "feedback_missing": bool(feedback_meta.get("feedback_missing", False)),
        }
        if log_full or not self._logged_body29_names:
            record["req_joint_pos_29"] = list(joint_pos)
            record["req_joint_vel_29"] = list(joint_vel)
            record["target_joint_pos_29"] = list(target_joint_pos_29)
            if raw_action_29:
                record["raw_action_29"] = list(raw_action_29)
            record["tracking_err_29"] = list(tracking_err)
        if not self._logged_body29_names:
            record["applied_body29_dof_names"] = list(self.sonic_joint_names_29)
            self._logged_body29_names = True
        self._telemetry_log(record)

        self._last_sonic_call_meta = {
            "step_idx": step_idx,
            "server_latency_ms": infer_latency_ms,
            "resp_max_abs_action": resp_max_abs_action,
            "raw_max_abs_action": raw_max_abs_action,
            "target_max_abs": target_max_abs,
            "delta_max_abs_before_clip": delta_before,
            "delta_max_abs_after_clip": delta_after,
            "policy_obs_dim": obs_dim,
            "policy_obs_used_dim": obs_used_dim,
            "policy_raw_action_dim": raw_action_dim,
        }
        if target_max_abs > 2.5:
            ref_perf = now_perf()
            dump_path = self._dump_recent_abort_window("direct_policy_abort", ref_perf)
            msg = (
                f"{self._locomotion_abort_token}: target_max_abs {target_max_abs:.3f} rad exceeded 2.5 rad safety gate."
            )
            if dump_path is not None:
                msg = f"{msg} Recent telemetry dump: {dump_path}"
            self._telemetry_log(
                {
                    "event": "direct_policy_abort",
                    "step_idx": step_idx,
                    "resp_max_abs_action": resp_max_abs_action,
                    "raw_max_abs_action": raw_max_abs_action,
                    "target_max_abs": target_max_abs,
                    "server_latency_ms": infer_latency_ms,
                    "actions_at_abort": list(target_joint_pos_29),
                    "raw_action_at_abort": list(raw_action_29) if raw_action_29 else None,
                    "error": msg,
                    "abort_dump": str(dump_path) if dump_path is not None else None,
                }
            )
            raise RuntimeError(msg)
        return stabilized

    def _request_sonic(self) -> List[float]:
        if self._sonic_client is None:
            self._init_sonic_client()
        assert self._sonic_client is not None

        step_idx = int(self._sonic_step_idx)
        self._sonic_step_idx += 1
        vx = float(self.sonic_velocity["vx"])
        vy = float(self.sonic_velocity["vy"])
        yaw_rate = float(self.sonic_velocity["yaw_rate"])
        style = str(self.sonic_velocity["style"])

        joint_pos, joint_vel, feedback_meta = self._build_sonic_request_from_live_feedback()
        if len(joint_pos) < 29:
            joint_pos.extend([0.0] * (29 - len(joint_pos)))
        if len(joint_vel) < 29:
            joint_vel.extend([0.0] * (29 - len(joint_vel)))
        joint_pos = [float(v) for v in joint_pos[:29]]
        joint_vel = [float(v) for v in joint_vel[:29]]
        req_is_all_zero_pos = self._is_all_zero(joint_pos, eps=1e-8)
        req_is_all_zero_vel = self._is_all_zero(joint_vel, eps=1e-8)
        req_max_abs_pos = self._max_abs(joint_pos)
        req_max_abs_vel = self._max_abs(joint_vel)

        if self._is_locomotion_command_active() and req_is_all_zero_pos:
            msg = (
                "SONIC_ABORT: locomotion command requested while req_joint_pos_29 is all zeros. "
                "Check live Isaac feedback mapping and joint-state stream."
            )
            self._telemetry_log(
                {
                    "event": "sonic_request_rejected",
                    "step_idx": step_idx,
                    "vx": vx,
                    "vy": vy,
                    "yaw_rate": yaw_rate,
                    "style": style,
                    "req_is_all_zero_pos": True,
                    "req_is_all_zero_vel": req_is_all_zero_vel,
                    "req_max_abs_pos": req_max_abs_pos,
                    "req_max_abs_vel": req_max_abs_vel,
                    "feedback_source": feedback_meta.get("feedback_source"),
                    "error": msg,
                }
            )
            raise RuntimeError(msg)

        t_req_send = None
        t_resp_recv = None
        try:
            resp = self._sonic_client.get_action(
                vx=vx,
                vy=vy,
                yaw_rate=yaw_rate,
                style=style,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
            )
            t_req_send, t_resp_recv = self._sonic_client.get_last_exchange_timing()
        except Exception:
            try:
                self._sonic_client.close()
            except Exception:
                pass
            self._sonic_client = None
            raise
        server_latency_ms = None
        if t_req_send is not None and t_resp_recv is not None:
            server_latency_ms = max(0.0, float(t_resp_recv - t_req_send) * 1000.0)
        if isinstance(resp, dict) and resp.get("error"):
            raise RuntimeError(f"SONIC server error: {resp['error']}")
        actions_payload = []
        raw_payload = []
        resp_info: Dict[str, Any] = {}
        if isinstance(resp, dict):
            actions_payload = resp.get("joint_actions") or []
            raw_payload = resp.get("raw_actions") or []
            info_payload = resp.get("info")
            if isinstance(info_payload, dict):
                resp_info = info_payload
        resp_joint_actions_29 = self._flatten_joint_payload_29(actions_payload)
        if not resp_joint_actions_29:
            raise RuntimeError(f"SONIC response did not include joint_actions: {resp}")
        raw_action_29 = self._flatten_joint_payload_29(raw_payload)
        target_joint_pos_29 = [float(v) for v in resp_joint_actions_29[:29]]
        raw_max_abs_action = self._max_abs(raw_action_29) if raw_action_29 else None
        target_max_abs = self._max_abs(target_joint_pos_29)
        # Keep historical field name for compatibility. Prefer target-space value for safety gates.
        resp_max_abs_action = float(raw_max_abs_action if raw_max_abs_action is not None else target_max_abs)

        stabilized = self._stabilize_sonic_targets(target_joint_pos_29, self.sonic_control_dt_s)
        delta_before = self._max_abs([target_joint_pos_29[i] - self._joint_estimate[i] for i in range(len(self._joint_estimate))])
        delta_after = self._max_abs([stabilized[i] - self._joint_estimate[i] for i in range(len(self._joint_estimate))])
        tracking_err = [stabilized[i] - joint_pos[i] for i in range(min(len(stabilized), len(joint_pos)))]
        tracking_stats = compute_stats(tracking_err)
        tracking_rms = tracking_stats.get("rms")
        base_pose = self._action_adapter.get_last_base_pose() or {}
        base_roll = float(base_pose.get("roll", 0.0))
        base_pitch = float(base_pose.get("pitch", 0.0))
        base_yaw = float(base_pose.get("yaw", 0.0))
        base_height = float(base_pose.get("z", 0.0))
        fall_flag = bool(
            (abs(base_roll) > 0.9)
            or (abs(base_pitch) > 0.9)
            or (base_height > 0.0 and base_height < 0.22)
        )
        slip_flag = bool(self._is_locomotion_command_active() and req_max_abs_vel < 1e-3 and req_max_abs_pos > 1e-4)

        log_full = (step_idx % self._telemetry_full_dump_every) == 0
        record: Dict[str, Any] = {
            "event": "sonic_exchange",
            "step_idx": step_idx,
            "t_req_send": t_req_send,
            "t_resp_recv": t_resp_recv,
            "vx": vx,
            "vy": vy,
            "yaw_rate": yaw_rate,
            "style": style,
            "req_is_all_zero_pos": req_is_all_zero_pos,
            "req_is_all_zero_vel": req_is_all_zero_vel,
            "req_max_abs_pos": req_max_abs_pos,
            "req_max_abs_vel": req_max_abs_vel,
            "resp_max_abs_action": resp_max_abs_action,
            "raw_max_abs_action": raw_max_abs_action,
            "target_max_abs": target_max_abs,
            "delta_max_abs_before_clip": delta_before,
            "delta_max_abs_after_clip": delta_after,
            "tracking_err_max_abs": self._max_abs(tracking_err),
            "tracking_err_rms": float(tracking_rms) if tracking_rms is not None else None,
            "base_roll": base_roll,
            "base_pitch": base_pitch,
            "base_yaw": base_yaw,
            "base_height": base_height,
            "fall_flag": fall_flag,
            "slip_flag": slip_flag,
            "server_latency_ms": server_latency_ms,
            "applied_dof_count": len(self.sonic_joint_names_29),
            "applied_target_len": len(stabilized),
            "feedback_source": feedback_meta.get("feedback_source"),
            "feedback_missing": bool(feedback_meta.get("feedback_missing", False)),
        }
        if log_full or not self._logged_body29_names:
            record["req_joint_pos_29"] = list(joint_pos)
            record["req_joint_vel_29"] = list(joint_vel)
            record["resp_joint_actions_29"] = list(resp_joint_actions_29)
            record["target_joint_pos_29"] = list(target_joint_pos_29)
            if raw_action_29:
                record["raw_action_29"] = list(raw_action_29)
            record["fb_joint_pos_29"] = list(joint_pos)
            record["fb_joint_vel_29"] = list(joint_vel)
            record["tracking_err_29"] = list(tracking_err)
        if not self._logged_body29_names:
            record["applied_body29_dof_names"] = list(self.sonic_joint_names_29)
            self._logged_body29_names = True
        if resp_info:
            if "action_scale" in resp_info:
                record["action_scale"] = resp_info.get("action_scale")
            if "default_angles_29" in resp_info:
                record["default_angles_29"] = resp_info.get("default_angles_29")
            if "action_scale_multiplier" in resp_info:
                record["action_scale_multiplier"] = resp_info.get("action_scale_multiplier")
        self._telemetry_log(record)

        self._last_sonic_call_meta = {
            "step_idx": step_idx,
            "server_latency_ms": server_latency_ms,
            "resp_max_abs_action": resp_max_abs_action,
            "raw_max_abs_action": raw_max_abs_action,
            "target_max_abs": target_max_abs,
            "delta_max_abs_before_clip": delta_before,
            "delta_max_abs_after_clip": delta_after,
        }
        if target_max_abs > 2.5:
            ref_perf = float(t_resp_recv if t_resp_recv is not None else now_perf())
            dump_path = self._dump_recent_abort_window("sonic_abort", ref_perf)
            msg = f"SONIC_ABORT: target_max_abs {target_max_abs:.3f} rad exceeded 2.5 rad safety gate."
            if dump_path is not None:
                msg = f"{msg} Recent telemetry dump: {dump_path}"
            self._telemetry_log(
                {
                    "event": "sonic_abort",
                    "step_idx": step_idx,
                    "resp_max_abs_action": resp_max_abs_action,
                    "raw_max_abs_action": raw_max_abs_action,
                    "target_max_abs": target_max_abs,
                    "server_latency_ms": server_latency_ms,
                    "actions_at_abort": list(target_joint_pos_29),
                    "raw_action_at_abort": list(raw_action_29) if raw_action_29 else None,
                    "error": msg,
                    "abort_dump": str(dump_path) if dump_path is not None else None,
                }
            )
            raise RuntimeError(msg)
        if raw_max_abs_action is not None and raw_max_abs_action > 2.5:
            self._telemetry_log(
                {
                    "event": "sonic_raw_spike",
                    "step_idx": step_idx,
                    "raw_max_abs_action": raw_max_abs_action,
                    "target_max_abs": target_max_abs,
                    "server_latency_ms": server_latency_ms,
                }
            )
        return stabilized

    def _stabilize_sonic_targets(self, target: List[float], dt_s: float) -> List[float]:
        target_vec = [float(v) for v in target]
        if len(target_vec) != len(self._joint_estimate):
            return target_vec

        prev = self._joint_estimate
        alpha = max(0.0, min(1.0, float(self.sonic_joint_blend_alpha)))
        if alpha < 1.0:
            blended = [float((1.0 - alpha) * prev[i] + alpha * target_vec[i]) for i in range(len(target_vec))]
        else:
            blended = target_vec

        max_rate = max(0.0, float(self.sonic_joint_rate_limit_rad_s))
        max_delta = max_rate * max(1e-3, float(dt_s))
        if max_delta <= 0.0:
            return blended

        limited: List[float] = []
        for i, cmd in enumerate(blended):
            d = float(cmd - prev[i])
            if d > max_delta:
                d = max_delta
            elif d < -max_delta:
                d = -max_delta
            limited.append(float(prev[i] + d))
        return limited

    def _load_modality_metadata(self) -> None:
        proc = self._load_model_json("processor_config.json")
        stats = self._load_model_json("statistics.json")
        emb_key = self._resolve_embodiment_key(proc, stats)

        mod_cfgs = proc.get("processor_kwargs", {}).get("modality_configs", {})
        emb_cfg = mod_cfgs.get(emb_key, {})
        self._video_keys = list(emb_cfg.get("video", {}).get("modality_keys", []))
        self._language_keys = list(emb_cfg.get("language", {}).get("modality_keys", []))
        self._video_horizon = max(1, len(emb_cfg.get("video", {}).get("delta_indices", []) or [0]))
        self._state_horizon = max(1, len(emb_cfg.get("state", {}).get("delta_indices", []) or [0]))

        emb_stats = stats.get(emb_key, {})
        self._state_dims = {
            key: len((vals.get("mean") or []))
            for key, vals in emb_stats.get("state", {}).items()
            if isinstance(vals, dict)
        }
        self._action_dims = {
            key: len((vals.get("mean") or []))
            for key, vals in emb_stats.get("action", {}).items()
            if isinstance(vals, dict)
        }

        if not self._video_keys:
            self._video_keys = ["ego_view"]
        if not self._state_dims:
            # Safe fallback for UNITREE_G1 when stats cannot be loaded.
            self._state_dims = {
                "left_leg": 6,
                "right_leg": 6,
                "waist": 3,
                "left_arm": 7,
                "right_arm": 7,
                "left_hand": 7,
                "right_hand": 7,
            }

        logging.info(
            "Loaded GR00T metadata: embodiment=%s video_keys=%s state_keys=%s action_keys=%s",
            emb_key,
            self._video_keys,
            sorted(self._state_dims.keys()),
            sorted(self._action_dims.keys()),
        )

    def _resolve_embodiment_key(self, proc: Dict[str, Any], stats: Dict[str, Any]) -> str:
        requested = self.embodiment_tag.strip().lower()
        proc_keys = set(proc.get("processor_kwargs", {}).get("modality_configs", {}).keys())
        stats_keys = set(stats.keys())
        all_keys = {k.lower(): k for k in (proc_keys | stats_keys)}

        if requested in all_keys:
            return all_keys[requested]
        if requested == "unitree_g1" and "unitree_g1" in all_keys:
            return all_keys["unitree_g1"]
        if "unitree_g1" in all_keys:
            logging.warning("Embodiment '%s' not found. Falling back to 'unitree_g1'.", self.embodiment_tag)
            return all_keys["unitree_g1"]
        if all_keys:
            fallback = sorted(all_keys.values())[0]
            logging.warning("Embodiment '%s' not found. Falling back to '%s'.", self.embodiment_tag, fallback)
            return fallback
        raise RuntimeError("No embodiment keys found in GR00T model metadata.")

    def _load_model_json(self, filename: str) -> Dict[str, Any]:
        if self.local_model_dir:
            path = Path(self.local_model_dir) / filename
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))

        if self.allow_remote_metadata and "/" in self.repo_id:
            repo_q = urllib.parse.quote(self.repo_id, safe="/")
            url = f"https://huggingface.co/{repo_q}/resolve/main/{filename}"
            with urllib.request.urlopen(url, timeout=self.metadata_timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))

        raise RuntimeError(
            f"Cannot load {filename}. Set manipulation.local_model_dir or enable remote metadata."
        )

    def _ensure_policy_server_running(self) -> None:
        if self._is_tcp_open(self.policy_host, self.policy_port, timeout_s=1.0):
            return
        if not self.policy_start_command:
            raise RuntimeError(
                "Policy server is not reachable and no start_command configured. "
                "Run Isaac-GR00T server manually or set manipulation.policy_server.start_command."
            )

        cwd = self.policy_start_cwd or None
        logging.info("Starting GR00T policy server: %s", self.policy_start_command)
        self._policy_process = subprocess.Popen(self.policy_start_command, cwd=cwd, shell=True)

        deadline = time.time() + self.policy_start_timeout_s
        while time.time() < deadline:
            if self._is_tcp_open(self.policy_host, self.policy_port, timeout_s=1.0):
                logging.info("GR00T policy server became reachable on %s:%s", self.policy_host, self.policy_port)
                return
            time.sleep(0.5)
        raise RuntimeError("Timed out waiting for GR00T policy server startup.")

    @staticmethod
    def _is_tcp_open(host: str, port: int, timeout_s: float) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                return True
        except Exception:
            return False

    def _build_observation(self, image: Optional[Any], instruction: str) -> Dict[str, Any]:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            raise RuntimeError("numpy is required for GR00T policy observation building.") from exc

        # Isaac-GR00T Gr00tSimPolicyWrapper expects a flat observation schema:
        #   video.<key>, state.<key>, <language_key>
        obs: Dict[str, Any] = {}
        rgb = self._prepare_rgb_image(image, self.image_size)
        for video_key in self._video_keys:
            video = np.expand_dims(rgb, axis=(0, 1))
            if self._video_horizon > 1:
                video = np.repeat(video, self._video_horizon, axis=1)
            obs[f"video.{video_key}"] = video

        for key, dim in self._state_dims.items():
            raw = self._last_state_vectors.get(key)
            if raw is None:
                vec = np.zeros((dim,), dtype=np.float32)
            else:
                vec = np.asarray(raw, dtype=np.float32).reshape(-1)
                if vec.shape[0] < dim:
                    pad = np.zeros((dim - vec.shape[0],), dtype=np.float32)
                    vec = np.concatenate([vec, pad], axis=0)
                elif vec.shape[0] > dim:
                    vec = vec[:dim]
            state = vec.reshape(1, 1, dim)
            if self._state_horizon > 1:
                state = np.repeat(state, self._state_horizon, axis=1)
            obs[f"state.{key}"] = state

        lang_keys = self._language_keys or ["task"]
        for key in lang_keys:
            # Sim wrapper expects shape (B,), where B=1.
            obs[key] = [instruction]

        return obs

    @staticmethod
    def _prepare_rgb_image(image: Optional[Any], size: int) -> Any:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            raise RuntimeError("numpy is required for GR00T image processing.") from exc

        if image is None:
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            return arr

        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3 or arr.shape[-1] != 3:
            arr = np.zeros((size, size, 3), dtype=np.uint8)

        if arr.dtype != np.uint8:
            arr = arr.astype(float)
            arr = arr.clip(0.0, 255.0).astype(np.uint8)

        h, w = arr.shape[:2]
        if h != size or w != size:
            y_idx = np.linspace(0, h - 1, size).astype(int)
            x_idx = np.linspace(0, w - 1, size).astype(int)
            arr = arr[y_idx][:, x_idx]
        return arr

    @staticmethod
    def _extract_action_steps(actions_payload: Dict[str, Any]) -> List[Dict[str, List[float]]]:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            raise RuntimeError("numpy is required for GR00T action parsing.") from exc

        arrays: Dict[str, Any] = {}
        horizon = 1
        for key, raw in actions_payload.items():
            normalized_key = key[7:] if key.startswith("action.") else key
            arr = np.asarray(raw, dtype=float)
            # Policy responses are typically (B, T, D). Collapse batch dimension when B=1.
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            arrays[normalized_key] = arr
            horizon = max(horizon, int(arr.shape[0]))

        steps: List[Dict[str, List[float]]] = []
        for t in range(horizon):
            step: Dict[str, List[float]] = {}
            for key, arr in arrays.items():
                idx = min(t, arr.shape[0] - 1)
                step[key] = [float(v) for v in arr[idx].tolist()]
            steps.append(step)
        return steps
