IMAGE_TOPIC_NAME = "realsense/color/image_raw"
STATE_TOPIC_NAME = "G1Env/env_state_act"
CONTROL_GOAL_TOPIC = "ControlPolicy/upper_body_pose"
ROBOT_CONFIG_TOPIC = "WBCPolicy/robot_config"
KEYBOARD_INPUT_TOPIC = "/keyboard_input"
LOCO_MANIP_TASK_STATUS_TOPIC = "LocoManipPolicy/task_status"
LOCO_NAV_TASK_STATUS_TOPIC = "NavigationPolicy/task_status"
LOWER_BODY_POLICY_STATUS_TOPIC = "ControlPolicy/lower_body_policy_status"
JOINT_SAFETY_STATUS_TOPIC = "ControlPolicy/joint_safety_status"

ISAAC_INTERNAL_STATE_TOPIC = "G1Env/isaac_state"
ISAAC_INTERNAL_COMMAND_TOPIC = "G1Env/isaac_joint_command"
ISAAC_JOINT_STATES_TOPIC = "/joint_states"
ISAAC_TF_TOPIC = "/tf"
ISAAC_CLOCK_TOPIC = "/clock"
ISAAC_IMU_TOPIC = "/imu"
ISAAC_COMMAND_TOPIC = "/isaac/joint_command"


DEFAULT_NAV_CMD = [0.0, 0.0, 0.0]
DEFAULT_BASE_HEIGHT = 0.74
DEFAULT_WRIST_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2  # x, y, z + w, x, y, z

DEFAULT_MODEL_SERVER_PORT = 5555  # port used to host the model server
