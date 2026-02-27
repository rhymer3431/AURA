from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

G1_LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
G1_WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
G1_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
G1_PD_JOINT_ORDER = G1_LEG_JOINTS + G1_WAIST_JOINTS + G1_ARM_JOINTS
# From GR00T-WholeBodyControl decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml
G1_PD_KP = [
    150.0,
    150.0,
    150.0,
    200.0,
    40.0,
    40.0,
    150.0,
    150.0,
    150.0,
    200.0,
    40.0,
    40.0,
    250.0,
    250.0,
    250.0,
    100.0,
    100.0,
    40.0,
    40.0,
    20.0,
    20.0,
    20.0,
    100.0,
    100.0,
    40.0,
    40.0,
    20.0,
    20.0,
    20.0,
]
G1_PD_KD = [
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    5.0,
    5.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
]
G1_PD_KP_BY_NAME = dict(zip(G1_PD_JOINT_ORDER, G1_PD_KP))
G1_PD_KD_BY_NAME = dict(zip(G1_PD_JOINT_ORDER, G1_PD_KD))


def _log_robot_dof_snapshot(robot) -> Dict[str, Any]:
    try:
        positions = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1)
    except Exception as exc:
        logging.warning("Failed to query joint positions for DOF snapshot: %s", exc)
        positions = np.zeros((int(robot.num_dof),), dtype=np.float32)
    dof_names = [str(name) for name in list(robot.dof_names)]
    logging.info("Isaac Sim G1 DOF count: %d", int(robot.num_dof))
    logging.info("DOF names in Isaac Sim order:")
    for idx, name in enumerate(dof_names):
        pos = float(positions[idx]) if idx < positions.shape[0] else 0.0
        logging.info("  [%02d] %s: %.4f rad", idx, name, pos)
    return {
        "num_dof": int(robot.num_dof),
        "dof_names": dof_names,
        "joint_positions": [float(v) for v in positions.tolist()],
    }


def _log_gain_summary(
    label: str, names: list[str], joint_names: list[str], kps: np.ndarray, kds: np.ndarray
) -> Optional[Dict[str, Any]]:
    indices = [joint_names.index(name) for name in names if name in joint_names]
    if not indices:
        return None
    g_kps = kps[indices]
    g_kds = kds[indices]
    summary = {
        "label": label,
        "joints": [joint_names[i] for i in indices],
        "joint_count": len(indices),
        "kp_min": float(np.min(g_kps)),
        "kp_max": float(np.max(g_kps)),
        "kd_min": float(np.min(g_kds)),
        "kd_max": float(np.max(g_kds)),
    }
    logging.info(
        "PD %-5s joints=%d kp[min=%.1f max=%.1f] kd[min=%.1f max=%.1f]",
        summary["label"],
        summary["joint_count"],
        summary["kp_min"],
        summary["kp_max"],
        summary["kd_min"],
        summary["kd_max"],
    )
    return summary


def _apply_g1_pd_gains(robot) -> Dict[str, Any]:
    joint_names = list(robot.dof_names)
    num_dof = int(robot.num_dof)
    kps = np.zeros((num_dof,), dtype=np.float32)
    kds = np.zeros((num_dof,), dtype=np.float32)
    matched = 0
    for idx, name in enumerate(joint_names):
        kp = G1_PD_KP_BY_NAME.get(name)
        kd = G1_PD_KD_BY_NAME.get(name)
        if kp is None or kd is None:
            continue
        kps[idx] = float(kp)
        kds[idx] = float(kd)
        matched += 1

    if matched == 0:
        logging.warning("No G1 body joints matched for PD gain application; skipping set_gains.")
        return {"matched_dof": 0, "num_dof": num_dof, "groups": [], "remaining_dof_count": num_dof}

    controller = robot.get_articulation_controller()
    try:
        controller.set_gains(kps=kps, kds=kds)
        logging.info("PD gains applied to %d/%d DOFs.", matched, num_dof)
    except Exception as exc:
        logging.warning("Failed to apply PD gains: %s", exc)
        return {
            "matched_dof": matched,
            "num_dof": num_dof,
            "groups": [],
            "remaining_dof_count": max(0, num_dof - matched),
        }

    group_summaries: list[Dict[str, Any]] = []
    try:
        applied_kps_raw, applied_kds_raw = controller.get_gains()
        applied_kps = np.asarray(applied_kps_raw, dtype=np.float32).reshape(-1)
        applied_kds = np.asarray(applied_kds_raw, dtype=np.float32).reshape(-1)
        legs = _log_gain_summary("legs", G1_LEG_JOINTS, joint_names, applied_kps, applied_kds)
        waist = _log_gain_summary("waist", G1_WAIST_JOINTS, joint_names, applied_kps, applied_kds)
        arms = _log_gain_summary("arms", G1_ARM_JOINTS, joint_names, applied_kps, applied_kds)
        for item in (legs, waist, arms):
            if item is not None:
                group_summaries.append(item)
    except Exception as exc:
        logging.warning("Could not verify PD gains via get_gains(): %s", exc)
    return {
        "matched_dof": matched,
        "num_dof": num_dof,
        "groups": group_summaries,
        "remaining_dof_count": max(0, num_dof - matched),
    }
