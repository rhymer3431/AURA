"""Shared constants for the standalone G1 locomotion runner."""

from __future__ import annotations


HEIGHT_SCAN_OFFSET = 0.5
HEIGHT_SCAN_RAY_START_Z = 20.0
HEIGHT_SCAN_RAYCAST_DISTANCE = 30.0
FEET_CONTACT_HISTORY_LENGTH = 3
FEET_CONTACT_THRESHOLD = 1.0
DEFAULT_PHYSICS_DT = 1.0 / 200.0
DEFAULT_DECIMATION = 4

LEGACY_ACTION_SCALE = 0.5
TUNED_ACTION_SCALE = 0.33

LEGACY_HEIGHT_SCAN_SIZE = (1.6, 1.0)
LEGACY_HEIGHT_SCAN_RESOLUTION = 0.1
TUNED_HEIGHT_SCAN_SIZE = (2.6, 1.6)
TUNED_HEIGHT_SCAN_RESOLUTION = 0.05

DEFAULT_JOINT_POS_PATTERNS = {
    ".*_hip_pitch_joint": -0.28,
    ".*_knee_joint": 0.63,
    ".*_ankle_pitch_joint": -0.35,
    ".*_elbow_pitch_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
    "left_one_joint": 1.0,
    "right_one_joint": -1.0,
}

LEGACY_STIFFNESS_PATTERNS = {
    ".*_hip_yaw_joint": 150.0,
    ".*_hip_roll_joint": 150.0,
    ".*_hip_pitch_joint": 200.0,
    ".*_knee_joint": 200.0,
    "torso_joint": 200.0,
    ".*_ankle_pitch_joint": 20.0,
    ".*_ankle_roll_joint": 20.0,
    ".*_shoulder_pitch_joint": 40.0,
    ".*_shoulder_roll_joint": 40.0,
    ".*_shoulder_yaw_joint": 40.0,
    ".*_elbow_pitch_joint": 40.0,
    ".*_elbow_roll_joint": 40.0,
    ".*_five_joint": 40.0,
    ".*_three_joint": 40.0,
    ".*_six_joint": 40.0,
    ".*_four_joint": 40.0,
    ".*_zero_joint": 40.0,
    ".*_one_joint": 40.0,
    ".*_two_joint": 40.0,
}

TUNED_STIFFNESS_PATTERNS = {
    ".*_hip_yaw_joint": 170.0,
    ".*_hip_roll_joint": 170.0,
    ".*_hip_pitch_joint": 240.0,
    ".*_knee_joint": 240.0,
    "torso_joint": 300.0,
    ".*_ankle_pitch_joint": 40.0,
    ".*_ankle_roll_joint": 40.0,
    ".*_shoulder_pitch_joint": 40.0,
    ".*_shoulder_roll_joint": 40.0,
    ".*_shoulder_yaw_joint": 40.0,
    ".*_elbow_pitch_joint": 40.0,
    ".*_elbow_roll_joint": 40.0,
    ".*_five_joint": 40.0,
    ".*_three_joint": 40.0,
    ".*_six_joint": 40.0,
    ".*_four_joint": 40.0,
    ".*_zero_joint": 40.0,
    ".*_one_joint": 40.0,
    ".*_two_joint": 40.0,
}

LEGACY_DAMPING_PATTERNS = {
    ".*_hip_yaw_joint": 5.0,
    ".*_hip_roll_joint": 5.0,
    ".*_hip_pitch_joint": 5.0,
    ".*_knee_joint": 5.0,
    "torso_joint": 5.0,
    ".*_ankle_pitch_joint": 4.0,
    ".*_ankle_roll_joint": 4.0,
    ".*_shoulder_pitch_joint": 10.0,
    ".*_shoulder_roll_joint": 10.0,
    ".*_shoulder_yaw_joint": 10.0,
    ".*_elbow_pitch_joint": 10.0,
    ".*_elbow_roll_joint": 10.0,
    ".*_five_joint": 10.0,
    ".*_three_joint": 10.0,
    ".*_six_joint": 10.0,
    ".*_four_joint": 10.0,
    ".*_zero_joint": 10.0,
    ".*_one_joint": 10.0,
    ".*_two_joint": 10.0,
}

TUNED_DAMPING_PATTERNS = {
    ".*_hip_yaw_joint": 7.0,
    ".*_hip_roll_joint": 7.0,
    ".*_hip_pitch_joint": 7.0,
    ".*_knee_joint": 7.0,
    "torso_joint": 10.0,
    ".*_ankle_pitch_joint": 8.0,
    ".*_ankle_roll_joint": 8.0,
    ".*_shoulder_pitch_joint": 10.0,
    ".*_shoulder_roll_joint": 10.0,
    ".*_shoulder_yaw_joint": 10.0,
    ".*_elbow_pitch_joint": 10.0,
    ".*_elbow_roll_joint": 10.0,
    ".*_five_joint": 10.0,
    ".*_three_joint": 10.0,
    ".*_six_joint": 10.0,
    ".*_four_joint": 10.0,
    ".*_zero_joint": 10.0,
    ".*_one_joint": 10.0,
    ".*_two_joint": 10.0,
}
