from __future__ import annotations

"""Deprecated: use apps.isaacsim_runner.control.g1 instead."""

from apps.isaacsim_runner.control.g1 import *  # noqa: F401, F403
from apps.isaacsim_runner.control.g1 import _log_gain_summary

G1_LEG_JOINTS = LEG_JOINTS
G1_WAIST_JOINTS = WAIST_JOINTS
G1_ARM_JOINTS = ARM_JOINTS
G1_PD_JOINT_ORDER = PD_JOINT_ORDER
G1_PD_KP = PD_KP
G1_PD_KD = PD_KD
G1_PD_KP_BY_NAME = PD_KP_BY_NAME
G1_PD_KD_BY_NAME = PD_KD_BY_NAME

_log_robot_dof_snapshot = log_robot_dof_snapshot
_apply_g1_pd_gains = apply_pd_gains
