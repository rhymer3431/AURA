#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Tuple

import msgpack
import numpy as np
import onnxruntime as ort
import zmq

try:
    from .telemetry_runtime import JsonlTelemetryLogger, now_perf
except Exception:
    try:
        from apps.sonic_policy_server.telemetry_runtime import JsonlTelemetryLogger, now_perf
    except Exception:
        try:
            from telemetry_runtime import JsonlTelemetryLogger, now_perf
        except Exception:  # pragma: no cover - optional telemetry dependency
            JsonlTelemetryLogger = None  # type: ignore

            def now_perf() -> float:
                return time.perf_counter()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("sonic_policy_server")


# Mapping arrays copied from GR00T-WholeBodyControl policy_parameters.hpp.
# NOTE: The variable names in that header are historically confusing.
# Here we use explicit names for runtime conversion clarity.
MUJO_TO_ISAAC = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)
ISAAC_TO_MUJO = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)
LOWER_BODY_MUJO_ORDER_IN_ISAAC_INDEX = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int64
)
WRIST_ISAAC_INDEX = np.array([23, 24, 25, 26, 27, 28], dtype=np.int64)

DEFAULT_ANGLES_MUJO = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10.0 * 2.0 * math.pi
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ
EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0


def _build_action_scale() -> np.ndarray:
    s_5020 = 0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020
    s_7520_14 = 0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14
    s_7520_22 = 0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22
    s_4010 = 0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010
    return np.array(
        [
            s_7520_22,
            s_7520_22,
            s_7520_14,
            s_7520_22,
            s_5020,
            s_5020,
            s_7520_22,
            s_7520_22,
            s_7520_14,
            s_7520_22,
            s_5020,
            s_5020,
            s_7520_14,
            s_5020,
            s_5020,
            s_5020,
            s_5020,
            s_5020,
            s_5020,
            s_5020,
            s_4010,
            s_4010,
            s_5020,
            s_5020,
            s_5020,
            s_5020,
            s_5020,
            s_4010,
            s_4010,
        ],
        dtype=np.float32,
    )


G1_ACTION_SCALE = _build_action_scale()
DEFAULT_ANGLES_ISAAC = DEFAULT_ANGLES_MUJO[ISAAC_TO_MUJO]

PLANNER_ALLOWED_PRED_NUM_TOKENS = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)
PLANNER_DEFAULT_HEIGHT = 0.78874


STYLE_TO_MODE = {
    "normal": 2,
    "run": 3,
    "stealth": 18,
    "happy": 23,
    "crawl": 8,
}


def _resolve_mode(style: str, speed: float) -> int:
    key = style.strip().lower()
    if speed < 1e-3:
        return 0
    return STYLE_TO_MODE.get(key, 2)


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return fallback.astype(np.float32)
    return (v / n).astype(np.float32)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _quat_normalize(q1)
    w2, x2, y2, z2 = _quat_normalize(q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    rot = _quat_to_rot_matrix(q)
    return rot @ np.asarray(v, dtype=np.float64)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0n = _quat_normalize(q0)
    q1n = _quat_normalize(q1)
    dot = float(np.dot(q0n, q1n))
    if dot < 0.0:
        q1n = -q1n
        dot = -dot
    if dot > 0.9995:
        out = q0n + t * (q1n - q0n)
        return _quat_normalize(out)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return s0 * q0n + s1 * q1n


def _reorder_mujo_to_isaac(joints_mujo: np.ndarray) -> np.ndarray:
    joints_mujo = np.asarray(joints_mujo, dtype=np.float32)
    if joints_mujo.ndim == 1:
        out = np.zeros((29,), dtype=np.float32)
        out[MUJO_TO_ISAAC] = joints_mujo
        return out
    out = np.zeros_like(joints_mujo)
    out[..., MUJO_TO_ISAAC] = joints_mujo
    return out


def _reorder_isaac_to_mujo(joints_isaac: np.ndarray) -> np.ndarray:
    joints_isaac = np.asarray(joints_isaac, dtype=np.float32)
    return joints_isaac[..., MUJO_TO_ISAAC]


def _resample_planner_to_50hz(
    qpos_30hz_mujo: np.ndarray, num_pred_frames: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_frames = max(1, min(int(num_pred_frames), int(qpos_30hz_mujo.shape[0])))
    qpos = qpos_30hz_mujo[:valid_frames].astype(np.float64, copy=False)
    seconds = float(valid_frames) / 30.0
    timesteps_50 = max(2, int(math.floor(seconds * 50.0)))

    root_pos = np.zeros((timesteps_50, 3), dtype=np.float32)
    root_quat = np.zeros((timesteps_50, 4), dtype=np.float32)
    joints_isaac = np.zeros((timesteps_50, 29), dtype=np.float32)

    for f50 in range(timesteps_50):
        t = float(f50) / 50.0
        f30 = t * 30.0
        f0 = int(math.floor(f30))
        f1 = min(f0 + 1, valid_frames - 1)
        w1 = f30 - float(f0)
        w0 = 1.0 - w1

        root_pos[f50, :] = (w0 * qpos[f0, 0:3] + w1 * qpos[f1, 0:3]).astype(np.float32)
        root_quat[f50, :] = _quat_slerp(qpos[f0, 3:7], qpos[f1, 3:7], w1).astype(np.float32)

        joints_mujo = (w0 * qpos[f0, 7:36] + w1 * qpos[f1, 7:36]).astype(np.float32)
        joints_isaac[f50, :] = _reorder_mujo_to_isaac(joints_mujo)

    joint_vel_isaac = np.zeros_like(joints_isaac)
    joint_vel_isaac[:-1, :] = (joints_isaac[1:, :] - joints_isaac[:-1, :]) * 50.0
    joint_vel_isaac[-1, :] = joint_vel_isaac[-2, :]

    return root_pos, root_quat, joints_isaac, joint_vel_isaac


def _sample_frames(arr: np.ndarray, num_frames: int, step: int) -> np.ndarray:
    if arr.shape[0] == 0:
        raise ValueError("Cannot sample from empty array.")
    idxs = [min(i * step, arr.shape[0] - 1) for i in range(num_frames)]
    return arr[np.asarray(idxs, dtype=np.int64)]


@dataclass
class SonicRuntimeState:
    context_qpos_mujo: np.ndarray
    heading_yaw: float
    last_raw_action_isaac: np.ndarray
    hist_base_ang: Deque[np.ndarray]
    hist_body_q: Deque[np.ndarray]
    hist_body_dq: Deque[np.ndarray]
    hist_last_actions: Deque[np.ndarray]
    hist_gravity: Deque[np.ndarray]


class SonicPolicyServer:
    def __init__(
        self,
        encoder: ort.InferenceSession,
        decoder: ort.InferenceSession,
        planner: ort.InferenceSession,
        action_scale: np.ndarray,
        action_scale_multiplier: float = 1.0,
        telemetry: Any = None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.planner = planner
        self.action_scale = np.asarray(action_scale, dtype=np.float32).reshape(29)
        self.action_scale_multiplier = float(action_scale_multiplier)
        self.telemetry = telemetry
        self.state = self._init_runtime_state()
        self._infer_step_idx = 0

        self.encoder_input_name = self.encoder.get_inputs()[0].name
        self.decoder_input_name = self.decoder.get_inputs()[0].name
        self.planner_input_names = {inp.name for inp in self.planner.get_inputs()}
        self.planner_output_names = [out.name for out in self.planner.get_outputs()]
        self._log_policy_constants()

    def _log_policy_constants(self) -> None:
        LOGGER.info(
            "default_angles_mujo: %s",
            np.array2string(DEFAULT_ANGLES_MUJO, precision=4, separator=","),
        )
        LOGGER.info(
            "default_angles_isaac: %s",
            np.array2string(DEFAULT_ANGLES_ISAAC, precision=4, separator=","),
        )
        LOGGER.info(
            "action_scale: %s",
            np.array2string(self.action_scale, precision=6, separator=","),
        )
        if self.telemetry is not None:
            self.telemetry.log(
                {
                    "event": "policy_constants",
                    "default_angles_29": DEFAULT_ANGLES_ISAAC.tolist(),
                    "action_scale": self.action_scale.tolist(),
                    "action_scale_multiplier": self.action_scale_multiplier,
                    "action_scale_is_per_joint": bool(self.action_scale.shape[0] == 29),
                }
            )

    @staticmethod
    def _init_runtime_state() -> SonicRuntimeState:
        context = np.zeros((4, 36), dtype=np.float32)
        context[:, 2] = PLANNER_DEFAULT_HEIGHT
        context[:, 3] = 1.0
        context[:, 7:36] = DEFAULT_ANGLES_MUJO[np.newaxis, :]
        zeros29 = np.zeros((29,), dtype=np.float32)
        zeros3 = np.zeros((3,), dtype=np.float32)
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        hist_base_ang: Deque[np.ndarray] = deque([zeros3.copy() for _ in range(10)], maxlen=10)
        hist_body_q: Deque[np.ndarray] = deque(
            [(-DEFAULT_ANGLES_ISAAC).astype(np.float32) for _ in range(10)], maxlen=10
        )
        hist_body_dq: Deque[np.ndarray] = deque([zeros29.copy() for _ in range(10)], maxlen=10)
        hist_last_actions: Deque[np.ndarray] = deque([zeros29.copy() for _ in range(10)], maxlen=10)
        hist_gravity: Deque[np.ndarray] = deque([gravity.copy() for _ in range(10)], maxlen=10)
        return SonicRuntimeState(
            context_qpos_mujo=context,
            heading_yaw=0.0,
            last_raw_action_isaac=zeros29,
            hist_base_ang=hist_base_ang,
            hist_body_q=hist_body_q,
            hist_body_dq=hist_body_dq,
            hist_last_actions=hist_last_actions,
            hist_gravity=hist_gravity,
        )

    def _build_planner_inputs(
        self, vx: float, vy: float, yaw_rate: float, style: str, joint_pos_isaac: np.ndarray
    ) -> Dict[str, np.ndarray]:
        speed = float(math.sqrt(vx * vx + vy * vy))
        mode = _resolve_mode(style, speed)
        target_vel = np.array([speed], dtype=np.float32)

        movement_dir = _normalize(np.array([vx, vy, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0]))

        # Integrate heading as a lightweight proxy when only yaw_rate is provided.
        self.state.heading_yaw += float(yaw_rate) * 0.05
        facing_dir = np.array(
            [math.cos(self.state.heading_yaw), math.sin(self.state.heading_yaw), 0.0], dtype=np.float32
        )
        if speed > 1e-3:
            facing_dir = movement_dir

        # Planner context is in MuJoCo order.
        joint_pos_mujo = _reorder_isaac_to_mujo(joint_pos_isaac)
        self.state.context_qpos_mujo[:, 7:36] = joint_pos_mujo[np.newaxis, :]

        planner_inputs: Dict[str, np.ndarray] = {
            "context_mujoco_qpos": self.state.context_qpos_mujo[np.newaxis, :, :].astype(np.float32),
            "target_vel": target_vel,
            "mode": np.array([mode], dtype=np.int64),
            "movement_direction": movement_dir[np.newaxis, :].astype(np.float32),
            "facing_direction": facing_dir[np.newaxis, :].astype(np.float32),
            "random_seed": np.array([1234], dtype=np.int64),
            "has_specific_target": np.array([[0]], dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": PLANNER_ALLOWED_PRED_NUM_TOKENS.copy(),
            "height": np.array([PLANNER_DEFAULT_HEIGHT], dtype=np.float32),
        }
        # Keep only inputs expected by the model build.
        return {k: v for k, v in planner_inputs.items() if k in self.planner_input_names}

    @staticmethod
    def _anchor_orientation(base_quat: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
        base_to_ref = _quat_mul(_quat_conjugate(base_quat), ref_quat)
        rot = _quat_to_rot_matrix(base_to_ref)
        return np.array(
            [rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]],
            dtype=np.float32,
        )

    def _build_encoder_obs(
        self,
        root_pos_50hz: np.ndarray,
        root_quat_50hz: np.ndarray,
        joints_isaac_50hz: np.ndarray,
        joint_vel_isaac_50hz: np.ndarray,
    ) -> np.ndarray:
        obs = np.zeros((1762,), dtype=np.float32)
        offset = 0

        def put(values: np.ndarray) -> None:
            nonlocal offset
            flat = np.asarray(values, dtype=np.float32).reshape(-1)
            obs[offset : offset + flat.size] = flat
            offset += flat.size

        sampled_step5_pos = _sample_frames(joints_isaac_50hz, 10, 5)  # (10, 29)
        sampled_step5_vel = _sample_frames(joint_vel_isaac_50hz, 10, 5)  # (10, 29)
        sampled_step1_pos = _sample_frames(joints_isaac_50hz, 10, 1)  # (10, 29)
        sampled_step5_root_pos = _sample_frames(root_pos_50hz, 10, 5)  # (10, 3)
        sampled_step5_root_quat = _sample_frames(root_quat_50hz, 10, 5)  # (10, 4)
        sampled_step1_root_quat = _sample_frames(root_quat_50hz, 10, 1)  # (10, 4)

        base_quat = root_quat_50hz[0]

        put(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))  # encoder_mode_4
        put(sampled_step5_pos)  # motion_joint_positions_10frame_step5
        put(sampled_step5_vel)  # motion_joint_velocities_10frame_step5
        put(sampled_step5_root_pos[:, 2])  # motion_root_z_position_10frame_step5
        put(np.array([root_pos_50hz[0, 2]], dtype=np.float32))  # motion_root_z_position
        put(self._anchor_orientation(base_quat, root_quat_50hz[0]))  # motion_anchor_orientation
        put(
            np.stack(
                [self._anchor_orientation(base_quat, q) for q in sampled_step5_root_quat], axis=0
            )
        )  # motion_anchor_orientation_10frame_step5
        put(sampled_step5_pos[:, LOWER_BODY_MUJO_ORDER_IN_ISAAC_INDEX])  # lowerbody positions
        put(sampled_step5_vel[:, LOWER_BODY_MUJO_ORDER_IN_ISAAC_INDEX])  # lowerbody velocities
        put(np.zeros((9,), dtype=np.float32))  # vr_3point_local_target
        put(np.zeros((12,), dtype=np.float32))  # vr_3point_local_orn_target
        put(np.zeros((720,), dtype=np.float32))  # smpl_joints_10frame_step1
        put(
            np.stack(
                [self._anchor_orientation(base_quat, q) for q in sampled_step1_root_quat], axis=0
            )
        )  # smpl_anchor_orientation_10frame_step1
        put(sampled_step1_pos[:, WRIST_ISAAC_INDEX])  # motion_joint_positions_wrists_10frame_step1

        if offset != 1762:
            raise RuntimeError(f"Encoder observation packing mismatch: expected 1762, got {offset}")
        return obs

    def _build_policy_obs(
        self,
        token_state: np.ndarray,
        joint_pos_isaac: np.ndarray,
        joint_vel_isaac: np.ndarray,
        base_quat: np.ndarray,
    ) -> np.ndarray:
        base_ang_vel = np.zeros((3,), dtype=np.float32)
        gravity = _quat_rotate(_quat_conjugate(base_quat), np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(
            np.float32
        )
        body_q = (joint_pos_isaac - DEFAULT_ANGLES_ISAAC).astype(np.float32)
        body_dq = joint_vel_isaac.astype(np.float32)

        self.state.hist_base_ang.append(base_ang_vel)
        self.state.hist_body_q.append(body_q)
        self.state.hist_body_dq.append(body_dq)
        self.state.hist_last_actions.append(self.state.last_raw_action_isaac.astype(np.float32))
        self.state.hist_gravity.append(gravity)

        obs = np.zeros((994,), dtype=np.float32)
        offset = 0

        def put(values: Iterable[np.ndarray]) -> None:
            nonlocal offset
            if isinstance(values, np.ndarray):
                flat = values.astype(np.float32).reshape(-1)
            else:
                flat = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in values], axis=0)
            obs[offset : offset + flat.size] = flat
            offset += flat.size

        put(token_state.reshape(-1))  # token_state (64)
        put(list(self.state.hist_base_ang))  # 10 x 3
        put(list(self.state.hist_body_q))  # 10 x 29
        put(list(self.state.hist_body_dq))  # 10 x 29
        put(list(self.state.hist_last_actions))  # 10 x 29
        put(list(self.state.hist_gravity))  # 10 x 3

        if offset != 994:
            raise RuntimeError(f"Policy observation packing mismatch: expected 994, got {offset}")
        return obs

    def infer(
        self,
        vx: float,
        vy: float,
        yaw_rate: float,
        style: str,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        step_idx = int(self._infer_step_idx)
        self._infer_step_idx += 1
        t_infer_start = now_perf()
        joint_pos_isaac = np.asarray(joint_pos, dtype=np.float32).reshape(29)
        joint_vel_isaac = np.asarray(joint_vel, dtype=np.float32).reshape(29)

        planner_inputs = self._build_planner_inputs(vx, vy, yaw_rate, style, joint_pos_isaac)
        planner_output = self.planner.run(None, planner_inputs)
        mujoco_qpos = np.asarray(planner_output[0], dtype=np.float32)
        num_pred_frames = int(np.asarray(planner_output[1]).reshape(-1)[0])

        # Keep first four predicted frames as planner context for temporal continuity.
        self.state.context_qpos_mujo = mujoco_qpos[0, :4, :].astype(np.float32, copy=True)

        root_pos_50hz, root_quat_50hz, joints_isaac_50hz, joint_vel_isaac_50hz = _resample_planner_to_50hz(
            mujoco_qpos[0], num_pred_frames
        )
        encoder_obs = self._build_encoder_obs(root_pos_50hz, root_quat_50hz, joints_isaac_50hz, joint_vel_isaac_50hz)
        token_state = self.encoder.run(None, {self.encoder_input_name: encoder_obs.reshape(1, -1)})[0]
        token_state = np.asarray(token_state, dtype=np.float32).reshape(64)

        policy_obs = self._build_policy_obs(token_state, joint_pos_isaac, joint_vel_isaac, root_quat_50hz[0])
        raw_action_isaac = self.decoder.run(None, {self.decoder_input_name: policy_obs.reshape(1, -1)})[0]
        raw_action_isaac = np.asarray(raw_action_isaac, dtype=np.float32).reshape(29)
        self.state.last_raw_action_isaac = raw_action_isaac

        # Match reference deployment conversion: action -> scaled delta -> absolute target (MuJoCo order).
        raw_action_mujo = _reorder_isaac_to_mujo(raw_action_isaac)
        target_mujo = DEFAULT_ANGLES_MUJO + raw_action_mujo * self.action_scale
        target_isaac = _reorder_mujo_to_isaac(target_mujo)
        infer_latency_ms = max(0.0, (now_perf() - t_infer_start) * 1000.0)
        if self.telemetry is not None:
            self.telemetry.log(
                {
                    "event": "inference",
                    "step_idx": step_idx,
                    "vx": float(vx),
                    "vy": float(vy),
                    "yaw_rate": float(yaw_rate),
                    "style": str(style),
                    "req_is_all_zero_pos": bool(np.all(np.abs(joint_pos_isaac) <= 1e-8)),
                    "req_is_all_zero_vel": bool(np.all(np.abs(joint_vel_isaac) <= 1e-8)),
                    "req_max_abs_pos": float(np.max(np.abs(joint_pos_isaac))),
                    "req_max_abs_vel": float(np.max(np.abs(joint_vel_isaac))),
                    "raw_action_29": raw_action_isaac.tolist(),
                    "resp_max_abs_action": float(np.max(np.abs(raw_action_isaac))),
                    "target_joint_pos_29": target_isaac.tolist(),
                    "target_max_abs": float(np.max(np.abs(target_isaac))),
                    "default_angles_29": DEFAULT_ANGLES_ISAAC.tolist(),
                    "action_scale": self.action_scale.tolist(),
                    "server_infer_latency_ms": infer_latency_ms,
                }
            )

        return {
            "joint_actions": target_isaac.reshape(1, 29).astype(np.float32),
            "raw_actions": raw_action_isaac.reshape(1, 29).astype(np.float32),
            "token_state": token_state.reshape(1, 64).astype(np.float32),
        }


def _dict_get(d: Dict[Any, Any], key: str, default: Any = None) -> Any:
    if key in d:
        return d[key]
    bkey = key.encode("utf-8")
    if bkey in d:
        return d[bkey]
    return default


def _decode_array_like(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, dict):
        # Local ndarray codec used by other runtime modules.
        if "__ndarray_class__" in value:
            payload = value.get("as_npy")
            if isinstance(payload, (bytes, bytearray)):
                return np.load(io.BytesIO(payload), allow_pickle=False)
        # Best-effort compatibility for msgpack_numpy payloads.
        nd_flag = _dict_get(value, "nd")
        shape = _dict_get(value, "shape")
        dtype_str = _dict_get(value, "type")
        data = _dict_get(value, "data")
        if nd_flag and shape is not None and dtype_str is not None and data is not None:
            try:
                dtype = np.dtype(dtype_str)
                shape_tuple = tuple(int(v) for v in shape)
                expected_bytes = int(np.prod(shape_tuple)) * int(dtype.itemsize)
                raw = bytes(data)
                if len(raw) >= expected_bytes:
                    return np.frombuffer(raw[:expected_bytes], dtype=dtype).reshape(shape_tuple)
                # Some environments ship incompatible msgpack_numpy builds that truncate payloads.
                return np.zeros(shape_tuple, dtype=dtype)
            except Exception:
                pass
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return np.zeros((29,), dtype=np.float32)


def load_session(path: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in available:
        preferred = ["CPUExecutionProvider"]
        LOGGER.warning(
            "CUDAExecutionProvider unavailable. Falling back to CPUExecutionProvider for %s", path
        )
    try:
        sess = ort.InferenceSession(path, providers=preferred)
        LOGGER.info("Loaded ONNX model: %s | providers=%s", path, sess.get_providers())
        return sess
    except Exception:
        if preferred != ["CPUExecutionProvider"]:
            LOGGER.exception("Failed to load with CUDA provider for %s. Retrying CPU-only.", path)
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            LOGGER.info("Loaded ONNX model with CPU provider: %s", path)
            return sess
        raise


def run_server(
    encoder_path: str,
    decoder_path: str,
    planner_path: str,
    host: str = "0.0.0.0",
    port: int = 5556,
    action_scale_multiplier: float = 1.0,
) -> None:
    encoder = load_session(encoder_path)
    decoder = load_session(decoder_path)
    planner = load_session(planner_path)
    scale = G1_ACTION_SCALE * max(0.0, float(action_scale_multiplier))
    include_response_constants = os.environ.get("AURA_SONIC_INCLUDE_CONSTANTS", "0").strip() == "1"
    telemetry = None
    if JsonlTelemetryLogger is not None:
        try:
            telemetry = JsonlTelemetryLogger(
                phase=os.environ.get("AURA_TELEMETRY_PHASE", "standing"),
                component="sonic_server",
            )
        except Exception as exc:
            LOGGER.warning("Failed to initialize telemetry logger: %s", exc)
    if action_scale_multiplier != 1.0:
        LOGGER.warning("Applying action scale multiplier: %.3f", action_scale_multiplier)
    server = SonicPolicyServer(
        encoder=encoder,
        decoder=decoder,
        planner=planner,
        action_scale=scale,
        action_scale_multiplier=action_scale_multiplier,
        telemetry=telemetry,
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    LOGGER.info("SONIC policy server listening on %s:%d", host, port)

    try:
        while True:
            raw = sock.recv()
            t_req_recv = now_perf()
            try:
                req = msgpack.unpackb(raw, raw=False, strict_map_key=False)
                if not isinstance(req, dict):
                    raise TypeError(f"Invalid request type: {type(req).__name__}")
                vx = float(_dict_get(req, "vx", 0.0))
                vy = float(_dict_get(req, "vy", 0.0))
                yaw_rate = float(_dict_get(req, "yaw_rate", 0.0))
                style = str(_dict_get(req, "style", "normal"))
                joint_pos = _decode_array_like(
                    _dict_get(req, "joint_pos", np.zeros((29,), dtype=np.float32))
                )
                joint_vel = _decode_array_like(
                    _dict_get(req, "joint_vel", np.zeros((29,), dtype=np.float32))
                )
                joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(29)
                joint_vel = np.asarray(joint_vel, dtype=np.float32).reshape(29)
                locomotion_cmd = (abs(vx) + abs(vy) + abs(yaw_rate)) > 1e-3
                joint_pos_all_zero = bool(np.all(np.abs(joint_pos) <= 1e-8))
                if locomotion_cmd and joint_pos_all_zero:
                    msg = "locomotion command rejected: joint_pos is all zeros"
                    if telemetry is not None:
                        telemetry.log(
                            {
                                "event": "request_rejected",
                                "vx": vx,
                                "vy": vy,
                                "yaw_rate": yaw_rate,
                                "style": style,
                                "req_is_all_zero_pos": True,
                                "req_max_abs_pos": float(np.max(np.abs(joint_pos))),
                                "error": msg,
                            }
                        )
                    resp = {"joint_actions": None, "raw_actions": None, "info": None, "error": msg}
                    sock.send(msgpack.packb(resp, use_bin_type=True))
                    continue
                out = server.infer(vx, vy, yaw_rate, style, joint_pos, joint_vel)
                t_resp_send = now_perf()
                resp = {
                    "joint_actions": out["joint_actions"].tolist(),
                    "raw_actions": out["raw_actions"].tolist(),
                    "info": {"style": style},
                    "error": None,
                }
                resp["info"]["action_scale_multiplier"] = float(action_scale_multiplier)
                resp["info"]["action_scale_is_per_joint"] = True
                if include_response_constants:
                    resp["info"]["default_angles_29"] = DEFAULT_ANGLES_ISAAC.tolist()
                    resp["info"]["action_scale"] = scale.tolist()
                if telemetry is not None:
                    telemetry.log(
                        {
                            "event": "request_served",
                            "vx": vx,
                            "vy": vy,
                            "yaw_rate": yaw_rate,
                            "style": style,
                            "req_is_all_zero_pos": joint_pos_all_zero,
                            "req_is_all_zero_vel": bool(np.all(np.abs(joint_vel) <= 1e-8)),
                            "req_max_abs_pos": float(np.max(np.abs(joint_pos))),
                            "req_max_abs_vel": float(np.max(np.abs(joint_vel))),
                            "server_loop_latency_ms": max(0.0, (t_resp_send - t_req_recv) * 1000.0),
                        }
                    )
            except Exception as exc:
                LOGGER.exception("Inference error")
                if telemetry is not None:
                    telemetry.log({"event": "inference_error", "error": str(exc)})
                resp = {"joint_actions": None, "raw_actions": None, "info": None, "error": str(exc)}
            sock.send(msgpack.packb(resp, use_bin_type=True))
    finally:
        sock.close(linger=0)
        ctx.term()
        if telemetry is not None:
            try:
                telemetry.flush()
                telemetry.close()
            except Exception:
                pass


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "apps" / "gear_sonic_deploy"


def _parse_args() -> argparse.Namespace:
    default_dir = _default_models_dir()
    parser = argparse.ArgumentParser(description="GEAR-SONIC ZMQ policy server")
    parser.add_argument("--encoder", default=str(default_dir / "model_encoder.onnx"), required=False)
    parser.add_argument("--decoder", default=str(default_dir / "model_decoder.onnx"), required=False)
    parser.add_argument("--planner", default=str(default_dir / "planner_sonic.onnx"), required=False)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument(
        "--action-scale-multiplier",
        type=float,
        default=0.10,
        help="Multiplier applied to per-joint action scales (use <1.0 for conservative motion).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_server(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        planner_path=args.planner,
        host=args.host,
        port=args.port,
        action_scale_multiplier=args.action_scale_multiplier,
    )
