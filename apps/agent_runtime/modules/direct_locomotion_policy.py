from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_DEFAULT_NEUTRAL_JOINTS_29 = np.asarray(
    [
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
    ],
    dtype=np.float32,
)

# left leg 6 + right leg 6 in canonical body_29 indices
_DEFAULT_OUTPUT_INDICES_12_TO_29 = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]


class DirectLocomotionPolicyRuntime:
    """Direct locomotion policy runtime for Isaac Lab policies (.onnx or TorchScript .pt)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.policy_path = Path(str(self.cfg.get("policy_path", "models/policy.onnx"))).expanduser().resolve()
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Direct locomotion policy not found: {self.policy_path}")

        self.output_mode = str(self.cfg.get("output_mode", "delta")).strip().lower()
        if self.output_mode not in {"delta", "absolute"}:
            self.output_mode = "delta"

        self.action_clip = float(self.cfg.get("action_clip", 1.0))
        self.target_abs_clip = float(self.cfg.get("target_abs_clip", 2.5))
        self.hold_unmapped_joints = bool(self.cfg.get("hold_unmapped_joints", True))

        self.control_dt_s = float(self.cfg.get("control_dt_s", 0.02))
        self.gait_frequency_hz = float(self.cfg.get("gait_frequency_hz", 1.5))

        self.obs_include_prev_action = bool(self.cfg.get("obs_include_prev_action", True))
        self.obs_include_clock = bool(self.cfg.get("obs_include_clock", True))
        self.obs_include_base_rpy = bool(self.cfg.get("obs_include_base_rpy", False))
        self.command_scale = self._parse_scale3(self.cfg.get("command_scale", [1.0, 1.0, 1.0]))
        self.joint_pos_scale = float(self.cfg.get("joint_pos_scale", 1.0))
        self.joint_vel_scale = float(self.cfg.get("joint_vel_scale", 1.0))

        self.default_joint_pos_29 = self._parse_default_joint_pos(self.cfg.get("default_joint_pos_29"))
        self.action_scale_29 = self._parse_action_scale_29(
            self.cfg.get("action_scale_29"),
            scalar_fallback=float(self.cfg.get("action_scale", 0.25)),
        )

        raw_indices = self.cfg.get("output_indices_29", _DEFAULT_OUTPUT_INDICES_12_TO_29)
        self.output_indices_29 = self._parse_output_indices(raw_indices)

        self.backend = ""
        self._obs_dim = int(self.cfg.get("obs_dim", 0))
        self._raw_action_dim_hint = int(self.cfg.get("action_dim", 0))
        self._phase = 0.0
        self._last_raw_action_29 = np.zeros((29,), dtype=np.float32)
        self._shape_warned = False

        self._ort_session = None
        self._torch_model = None
        self._torch = None
        self._torch_device = "cpu"

        self._input_name = "obs"
        self._output_name = "actions"

        suffix = self.policy_path.suffix.lower()
        if suffix in {".pt", ".pth", ".jit", ".ts"}:
            self._init_torchscript()
        else:
            self._init_onnx()

        logging.info(
            "Direct locomotion policy loaded: backend=%s path=%s obs_dim=%s action_dim=%s mode=%s",
            self.backend,
            self.policy_path,
            self._obs_dim,
            self._raw_action_dim_hint,
            self.output_mode,
        )

    @staticmethod
    def _parse_scale3(raw: Any) -> np.ndarray:
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            try:
                return np.asarray([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float32)
            except Exception:
                pass
        try:
            v = float(raw)
            return np.asarray([v, v, v], dtype=np.float32)
        except Exception:
            return np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    @staticmethod
    def _parse_default_joint_pos(raw: Any) -> np.ndarray:
        if isinstance(raw, (list, tuple)) and len(raw) >= 29:
            try:
                arr = np.asarray(raw, dtype=np.float32).reshape(-1)
                return arr[:29]
            except Exception:
                pass
        return _DEFAULT_NEUTRAL_JOINTS_29.copy()

    @staticmethod
    def _parse_action_scale_29(raw: Any, scalar_fallback: float) -> np.ndarray:
        if isinstance(raw, (list, tuple)) and len(raw) >= 29:
            try:
                arr = np.asarray(raw, dtype=np.float32).reshape(-1)
                return np.maximum(np.abs(arr[:29]), 1e-6)
            except Exception:
                pass
        s = max(1e-6, abs(float(scalar_fallback)))
        return np.full((29,), s, dtype=np.float32)

    @staticmethod
    def _parse_output_indices(raw: Any) -> List[int]:
        indices: List[int] = []
        if isinstance(raw, (list, tuple)):
            for v in raw:
                try:
                    idx = int(v)
                except Exception:
                    continue
                if 0 <= idx < 29 and idx not in indices:
                    indices.append(idx)
        if not indices:
            indices = list(_DEFAULT_OUTPUT_INDICES_12_TO_29)
        return indices

    @staticmethod
    def _resolve_last_dim(shape: Any, fallback: int) -> int:
        if isinstance(shape, (list, tuple)) and shape:
            last = shape[-1]
            if isinstance(last, int) and last > 0:
                return int(last)
        if fallback > 0:
            return int(fallback)
        raise RuntimeError(f"Cannot resolve tensor last dimension from shape={shape}")

    def _init_onnx(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            raise RuntimeError("onnxruntime is required for .onnx direct policy.") from exc

        available = list(ort.get_available_providers())
        preferred_raw = self.cfg.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        preferred = [str(v) for v in (preferred_raw if isinstance(preferred_raw, (list, tuple)) else [preferred_raw])]
        providers = [p for p in preferred if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available

        session = ort.InferenceSession(str(self.policy_path), providers=providers)
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        if not inputs or not outputs:
            raise RuntimeError("Invalid ONNX policy: missing input/output tensors.")

        self._ort_session = session
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._obs_dim = self._resolve_last_dim(inputs[0].shape, fallback=self._obs_dim)
        self._raw_action_dim_hint = self._resolve_last_dim(outputs[0].shape, fallback=self._raw_action_dim_hint)
        self.backend = "onnx"

    def _init_torchscript(self) -> None:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError("torch is required for .pt/.pth direct policy.") from exc

        device = str(self.cfg.get("torch_device", "cpu"))
        model = torch.jit.load(str(self.policy_path), map_location=device)
        model.eval()

        self._torch = torch
        self._torch_model = model
        self._torch_device = device
        self.backend = "torchscript"

        if self._obs_dim <= 0:
            # Isaac Lab export default from the provided model archive.
            self._obs_dim = 310

        # Probe action dim once if possible.
        if self._raw_action_dim_hint <= 0:
            try:
                with torch.no_grad():
                    probe = torch.zeros((1, self._obs_dim), dtype=torch.float32, device=device)
                    out = model(probe)
                    if isinstance(out, (list, tuple)) and out:
                        out = out[0]
                    out_np = out.detach().cpu().numpy().reshape(-1)
                    self._raw_action_dim_hint = int(out_np.shape[0])
            except Exception:
                self._raw_action_dim_hint = 37

    def close(self) -> None:
        self._ort_session = None
        self._torch_model = None
        self._torch = None

    def _build_obs(
        self,
        vx: float,
        vy: float,
        yaw_rate: float,
        joint_pos_29: np.ndarray,
        joint_vel_29: np.ndarray,
        base_pose: Optional[Dict[str, float]],
    ) -> tuple[np.ndarray, int]:
        pos = np.asarray(joint_pos_29, dtype=np.float32).reshape(29)
        vel = np.asarray(joint_vel_29, dtype=np.float32).reshape(29)

        chunks: List[np.ndarray] = []
        cmd = np.asarray([vx, vy, yaw_rate], dtype=np.float32) * self.command_scale
        chunks.append(cmd)
        chunks.append((pos - self.default_joint_pos_29) * self.joint_pos_scale)
        chunks.append(vel * self.joint_vel_scale)

        if self.obs_include_prev_action:
            chunks.append(self._last_raw_action_29.copy())

        if self.obs_include_base_rpy:
            bp = base_pose or {}
            chunks.append(
                np.asarray(
                    [
                        float(bp.get("roll", 0.0)),
                        float(bp.get("pitch", 0.0)),
                        float(bp.get("yaw", 0.0)),
                    ],
                    dtype=np.float32,
                )
            )

        if self.obs_include_clock:
            chunks.append(
                np.asarray(
                    [math.sin(self._phase), math.cos(self._phase)],
                    dtype=np.float32,
                )
            )

        obs = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        used_dim = int(obs.shape[0])

        if used_dim < self._obs_dim:
            pad = np.zeros((self._obs_dim - used_dim,), dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=0)
        elif used_dim > self._obs_dim:
            obs = obs[: self._obs_dim]

        if used_dim != self._obs_dim and not self._shape_warned:
            self._shape_warned = True
            logging.warning(
                "Direct policy obs dim mismatch: constructed=%d model=%d (pad/truncate applied).",
                used_dim,
                self._obs_dim,
            )
        return obs.reshape(1, self._obs_dim), used_dim

    def _run_model(self, obs_batch: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            assert self._ort_session is not None
            out = self._ort_session.run([self._output_name], {self._input_name: obs_batch})[0]
            return np.asarray(out, dtype=np.float32).reshape(-1)

        if self.backend == "torchscript":
            assert self._torch is not None and self._torch_model is not None
            torch = self._torch
            with torch.no_grad():
                inp = torch.from_numpy(obs_batch).to(self._torch_device)
                out = self._torch_model(inp)
                if isinstance(out, (list, tuple)) and out:
                    out = out[0]
                return np.asarray(out.detach().cpu().numpy(), dtype=np.float32).reshape(-1)

        raise RuntimeError("Direct policy backend is not initialized.")

    def _map_raw_to_29(self, raw_action: np.ndarray) -> tuple[np.ndarray, List[int]]:
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        mapped = np.zeros((29,), dtype=np.float32)

        if raw.shape[0] >= 29:
            mapped[:] = raw[:29]
            return mapped, list(range(29))

        if raw.shape[0] == len(self.output_indices_29):
            indices = list(self.output_indices_29)
            for i, idx in enumerate(indices):
                mapped[idx] = raw[i]
            return mapped, indices

        n = min(int(raw.shape[0]), 29)
        if n > 0:
            mapped[:n] = raw[:n]
        return mapped, list(range(n))

    def infer(
        self,
        vx: float,
        vy: float,
        yaw_rate: float,
        joint_pos_29: np.ndarray,
        joint_vel_29: np.ndarray,
        base_pose: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        pos = np.asarray(joint_pos_29, dtype=np.float32).reshape(29)
        vel = np.asarray(joint_vel_29, dtype=np.float32).reshape(29)

        obs_batch, used_dim = self._build_obs(vx, vy, yaw_rate, pos, vel, base_pose)
        raw = self._run_model(obs_batch)
        raw_dim = int(raw.shape[0])

        raw_29, mapped_indices = self._map_raw_to_29(raw)
        clipped_29 = np.clip(raw_29, -self.action_clip, self.action_clip)

        if self.hold_unmapped_joints:
            target = pos.copy()
        else:
            target = self.default_joint_pos_29.copy()

        if self.output_mode == "absolute":
            for idx in mapped_indices:
                target[idx] = raw_29[idx]
        else:
            for idx in mapped_indices:
                target[idx] = self.default_joint_pos_29[idx] + clipped_29[idx] * self.action_scale_29[idx]

        if self.target_abs_clip > 0.0:
            target = np.clip(target, -self.target_abs_clip, self.target_abs_clip)

        self._last_raw_action_29 = raw_29
        self._phase += 2.0 * math.pi * self.gait_frequency_hz * self.control_dt_s
        if self._phase > 2.0 * math.pi:
            self._phase -= 2.0 * math.pi

        infer_latency_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
        return {
            "joint_actions_29": target.astype(np.float32),
            "raw_actions_29": raw_29.astype(np.float32),
            "raw_action_dim": raw_dim,
            "mapped_indices": mapped_indices,
            "obs_dim": int(self._obs_dim),
            "obs_used_dim": int(used_dim),
            "infer_latency_ms": float(infer_latency_ms),
        }

    @property
    def obs_dim(self) -> int:
        return int(self._obs_dim)

    @property
    def raw_action_dim(self) -> int:
        return int(self._raw_action_dim_hint)
