
import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import magnum as mn
import numpy as np

import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.visualizations.utils import observations_to_image, overlay_frame
from habitat_sim.utils import viz_utils as vut

try:
    import glfw
except ImportError:
    glfw = None

try:
    import OpenGL.GL as gl
except ImportError:
    gl = None


# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"


def step_env(env, action_name, action_args):
    return env.step({"action": action_name, "action_args": action_args})


class GLFWInput:
    """
    GLFW 키 상태 관리:
    - down(key): 현재 프레임에서 눌려있는지
    - pressed_once(key): 이번 프레임에 새로 눌렸는지(토글 키용)
    """

    def __init__(self, window):
        self.window = window
        self._prev_pressed: Set[int] = set()
        self._curr_pressed: Set[int] = set()

        # 실제로 사용하는 키만 추적
        self._keys_to_track = [
            glfw.KEY_ESCAPE,
            glfw.KEY_M,
            glfw.KEY_N,
            glfw.KEY_X,
            glfw.KEY_C,
            glfw.KEY_G,
            glfw.KEY_Z,
            glfw.KEY_B,
            glfw.KEY_I,
            glfw.KEY_J,
            glfw.KEY_K,
            glfw.KEY_L,
            glfw.KEY_U,
            glfw.KEY_O,
            glfw.KEY_W,
            glfw.KEY_A,
            glfw.KEY_S,
            glfw.KEY_D,
            glfw.KEY_Q,
            glfw.KEY_E,
            glfw.KEY_P,
            glfw.KEY_PERIOD,
            glfw.KEY_COMMA,
            glfw.KEY_1,
            glfw.KEY_2,
            glfw.KEY_3,
            glfw.KEY_4,
            glfw.KEY_5,
            glfw.KEY_6,
            glfw.KEY_7,
            glfw.KEY_R,
            glfw.KEY_T,
            glfw.KEY_Y,
        ]

    def poll(self):
        glfw.poll_events()
        self._prev_pressed = self._curr_pressed
        curr = set()
        for k in self._keys_to_track:
            if glfw.get_key(self.window, k) == glfw.PRESS:
                curr.add(k)
        self._curr_pressed = curr

    def down(self, key: int) -> bool:
        return key in self._curr_pressed

    def pressed_once(self, key: int) -> bool:
        return (key in self._curr_pressed) and (key not in self._prev_pressed)


class GLFWRenderer:
    """
    Habitat에서 만들어진 RGB 프레임(H, W, 3, uint8)을
    GLFW 창에 OpenGL 텍스처로 그려줌.
    """

    def __init__(self, width: int, height: int, title: str = "Habitat Interactive Play"):
        if glfw is None:
            raise ImportError("glfw is not installed. Run `pip install glfw`.")
        if gl is None:
            raise ImportError("PyOpenGL is not installed. Run `pip install PyOpenGL`.")

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # 고정 파이프라인 함수(glBegin/glOrtho) 사용 위해 2.1 힌트
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

        self.window = glfw.create_window(width, height, title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync

        self.width = width
        self.height = height
        self._tex_id = gl.glGenTextures(1)

        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        self._setup_projection(width, height)
        self._alloc_texture(width, height)

    def _setup_projection(self, width: int, height: int):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def _alloc_texture(self, width: int, height: int):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            width,
            height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )

    def should_close(self) -> bool:
        return glfw.window_should_close(self.window)

    def render(self, frame_rgb: np.ndarray):
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
            raise ValueError(f"Expected frame shape (H, W, >=3), got {frame_rgb.shape}")

        h, w = frame_rgb.shape[:2]
        if (w != self.width) or (h != self.height):
            self.width, self.height = w, h
            glfw.set_window_size(self.window, w, h)
            self._setup_projection(w, h)
            self._alloc_texture(w, h)

        frame = frame_rgb[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # OpenGL 좌표계(원점 하단) 맞춤
        frame = np.ascontiguousarray(np.flipud(frame))

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_id)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            frame,
        )

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex2f(-1.0, -1.0)

        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex2f(1.0, -1.0)

        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex2f(1.0, 1.0)

        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex2f(-1.0, 1.0)
        gl.glEnd()

        glfw.swap_buffers(self.window)

    def close(self):
        try:
            if self._tex_id is not None:
                gl.glDeleteTextures([self._tex_id])
        except Exception:
            pass

        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None

        glfw.terminate()


def get_input_vel_ctlr(
    skip_render,
    cfg,
    arm_action,
    env,
    not_block_input,
    agent_to_control,
    control_humanoid,
    humanoid_controller,
    kb: Optional[GLFWInput],
):
    if skip_render:
        # 렌더 없는 경우 기존 동작과 동일하게 empty step
        return step_env(env, "empty", {}), arm_action, False

    if kb is None:
        raise RuntimeError("GLFWInput is required when rendering is enabled.")

    multi_agent = len(env._sim.agents_mgr) > 1
    agent_k = f"agent_{agent_to_control}_" if multi_agent else ""

    arm_action_name = f"{agent_k}arm_action"

    # 기본 키 이름
    arm_key = "arm_action"
    grip_key = "grip_action"
    base_key = "base_vel"

    if control_humanoid:
        base_action_name = f"{agent_k}humanoidjoint_action"
        base_key = "human_joints_trans"
    else:
        if "spot" in cfg:
            base_action_name = f"{agent_k}base_velocity_non_cylinder"
        else:
            base_action_name = f"{agent_k}base_velocity"

    if arm_action_name in env.action_space.spaces:
        arm_action_space = env.action_space.spaces[arm_action_name].spaces[arm_key]
        arm_ctrlr = env.task.actions[arm_action_name].arm_ctrlr
        base_action = None
    elif "stretch" in cfg:
        arm_action_space = np.zeros(10)
        arm_ctrlr = None
        base_action = [0, 0]
    else:
        arm_action_space = np.zeros(7)
        arm_ctrlr = None
        base_action = [0, 0]

    if arm_action is None:
        arm_action = np.zeros(arm_action_space.shape[0])
        given_arm_action = False
    else:
        given_arm_action = True

    end_ep = False
    magic_grasp = None

    if kb.down(glfw.KEY_ESCAPE):
        return None, None, False
    elif kb.down(glfw.KEY_M):
        end_ep = True
    elif kb.pressed_once(glfw.KEY_N):
        env._sim.navmesh_visualization = not env._sim.navmesh_visualization

    if not_block_input:
        # Base control
        if kb.down(glfw.KEY_J):
            base_action = [0, 1]  # Left
        elif kb.down(glfw.KEY_L):
            base_action = [0, -1]  # Right
        elif kb.down(glfw.KEY_K):
            base_action = [-1, 0]  # Back
        elif kb.down(glfw.KEY_I):
            base_action = [1, 0]  # Forward

        if arm_action_space.shape[0] == 7:
            # Velocity control (7 joints)
            if kb.down(glfw.KEY_Q):
                arm_action[0] = 1.0
            elif kb.down(glfw.KEY_1):
                arm_action[0] = -1.0

            elif kb.down(glfw.KEY_W):
                arm_action[1] = 1.0
            elif kb.down(glfw.KEY_2):
                arm_action[1] = -1.0

            elif kb.down(glfw.KEY_E):
                arm_action[2] = 1.0
            elif kb.down(glfw.KEY_3):
                arm_action[2] = -1.0

            elif kb.down(glfw.KEY_R):
                arm_action[3] = 1.0
            elif kb.down(glfw.KEY_4):
                arm_action[3] = -1.0

            elif kb.down(glfw.KEY_T):
                arm_action[4] = 1.0
            elif kb.down(glfw.KEY_5):
                arm_action[4] = -1.0

            elif kb.down(glfw.KEY_Y):
                arm_action[5] = 1.0
            elif kb.down(glfw.KEY_6):
                arm_action[5] = -1.0

            elif kb.down(glfw.KEY_U):
                arm_action[6] = 1.0
            elif kb.down(glfw.KEY_7):
                arm_action[6] = -1.0

        elif arm_action_space.shape[0] == 4:
            # Spot arm (4 DoF)
            if kb.down(glfw.KEY_Q):
                arm_action[0] = 1.0
            elif kb.down(glfw.KEY_1):
                arm_action[0] = -1.0

            elif kb.down(glfw.KEY_W):
                arm_action[1] = 1.0
            elif kb.down(glfw.KEY_2):
                arm_action[1] = -1.0

            elif kb.down(glfw.KEY_E):
                arm_action[2] = 1.0
            elif kb.down(glfw.KEY_3):
                arm_action[2] = -1.0

            elif kb.down(glfw.KEY_R):
                arm_action[3] = 1.0
            elif kb.down(glfw.KEY_4):
                arm_action[3] = -1.0

        elif arm_action_space.shape[0] == 10:
            # Stretch arm style mapping
            if kb.down(glfw.KEY_Q):
                arm_action[0] = 1.0
            elif kb.down(glfw.KEY_1):
                arm_action[0] = -1.0

            elif kb.down(glfw.KEY_W):
                arm_action[4] = 1.0
            elif kb.down(glfw.KEY_2):
                arm_action[4] = -1.0

            elif kb.down(glfw.KEY_E):
                arm_action[5] = 1.0
            elif kb.down(glfw.KEY_3):
                arm_action[5] = -1.0

            elif kb.down(glfw.KEY_R):
                arm_action[6] = 1.0
            elif kb.down(glfw.KEY_4):
                arm_action[6] = -1.0

            elif kb.down(glfw.KEY_T):
                arm_action[7] = 1.0
            elif kb.down(glfw.KEY_5):
                arm_action[7] = -1.0

            elif kb.down(glfw.KEY_Y):
                arm_action[8] = 1.0
            elif kb.down(glfw.KEY_6):
                arm_action[8] = -1.0

            elif kb.down(glfw.KEY_U):
                arm_action[9] = 1.0
            elif kb.down(glfw.KEY_7):
                arm_action[9] = -1.0

        elif isinstance(arm_ctrlr, ArmEEAction):
            EE_FACTOR = 0.5
            # End effector control
            if kb.down(glfw.KEY_D):
                arm_action[1] -= EE_FACTOR
            elif kb.down(glfw.KEY_A):
                arm_action[1] += EE_FACTOR
            elif kb.down(glfw.KEY_W):
                arm_action[0] += EE_FACTOR
            elif kb.down(glfw.KEY_S):
                arm_action[0] -= EE_FACTOR
            elif kb.down(glfw.KEY_Q):
                arm_action[2] += EE_FACTOR
            elif kb.down(glfw.KEY_E):
                arm_action[2] -= EE_FACTOR
        else:
            raise ValueError("Unrecognized arm action space")

        if kb.down(glfw.KEY_P):
            logger.info("[play.py]: Unsnapping")
            magic_grasp = -1
        elif kb.down(glfw.KEY_O):
            logger.info("[play.py]: Snapping")
            magic_grasp = 1

    if control_humanoid:
        if humanoid_controller is None:
            # Add random noise to human arms but keep global transform
            joint_trans, root_trans = env._sim.articulated_agent.get_joint_transform()
            num_joints = len(joint_trans) // 4
            root_trans = np.array(root_trans)
            index_arms_start = 10
            joint_trans_quat = [
                mn.Quaternion(
                    mn.Vector3(joint_trans[(4 * index): (4 * index + 3)]),
                    joint_trans[4 * index + 3],
                )
                for index in range(num_joints)
            ]
            rotated_joints_quat = []
            for index, joint_quat in enumerate(joint_trans_quat):
                random_vec = np.random.rand(3)
                random_angle = np.random.rand() * 10
                rotation_quat = mn.Quaternion.rotation(
                    mn.Rad(random_angle), mn.Vector3(random_vec).normalized()
                )
                if index > index_arms_start:
                    joint_quat *= rotation_quat
                rotated_joints_quat.append(joint_quat)
            joint_trans = np.concatenate(
                [np.array(list(quat.vector) + [quat.scalar]) for quat in rotated_joints_quat]
            )
            base_action = np.concatenate(
                [joint_trans.reshape(-1), root_trans.transpose().reshape(-1)]
            )
        else:
            relative_pos = mn.Vector3(base_action[0], 0, base_action[1])
            humanoid_controller.calculate_walk_pose(relative_pos)
            base_action = humanoid_controller.get_pose()

    if kb.down(glfw.KEY_PERIOD):
        # Print articulated agent base state
        pos = [float("%.3f" % x) for x in env._sim.articulated_agent.sim_obj.translation]
        rot = env._sim.articulated_agent.sim_obj.rotation
        ee_pos = env._sim.articulated_agent.ee_transform().translation
        logger.info(f"Agent state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}")

    elif kb.down(glfw.KEY_COMMA):
        # Print arm joint state
        joint_state = [float("%.3f" % x) for x in env._sim.articulated_agent.arm_joint_pos]
        logger.info(f"Agent arm joint state: {joint_state}")

    args: Dict[str, Any] = {}

    if base_action is not None and base_action_name in env.action_space.spaces:
        name = base_action_name
        args = {base_key: base_action}
    else:
        name = arm_action_name
        if given_arm_action:
            # If action was loaded externally, include grip in tail
            args = {
                arm_key: arm_action[:-1],
                grip_key: arm_action[-1],
            }
        else:
            args = {arm_key: arm_action, grip_key: magic_grasp}

    if magic_grasp is None:
        arm_action = [*arm_action, 0.0]
    else:
        arm_action = [*arm_action, magic_grasp]

    return step_env(env, name, args), arm_action, end_ep


def get_wrapped_prop(venv, prop):
    if hasattr(venv, prop):
        return getattr(venv, prop)
    elif hasattr(venv, "venv"):
        return get_wrapped_prop(venv.venv, prop)
    elif hasattr(venv, "env"):
        return get_wrapped_prop(venv.env, prop)

    return None


class FreeCamHelper:
    def __init__(self):
        self._is_free_cam_mode = False
        self._free_rpy = np.zeros(3)
        self._free_xyz = np.zeros(3)

    @property
    def is_free_cam_mode(self):
        return self._is_free_cam_mode

    def update(self, env, step_result, kb: GLFWInput):
        if kb.pressed_once(glfw.KEY_Z):
            self._is_free_cam_mode = not self._is_free_cam_mode
            logger.info(f"Switching camera mode to {self._is_free_cam_mode}")

        if self._is_free_cam_mode:
            offset_rpy = np.zeros(3)
            if kb.down(glfw.KEY_U):
                offset_rpy[1] += 1
            elif kb.down(glfw.KEY_O):
                offset_rpy[1] -= 1
            elif kb.down(glfw.KEY_I):
                offset_rpy[2] += 1
            elif kb.down(glfw.KEY_K):
                offset_rpy[2] -= 1
            elif kb.down(glfw.KEY_J):
                offset_rpy[0] += 1
            elif kb.down(glfw.KEY_L):
                offset_rpy[0] -= 1

            offset_xyz = np.zeros(3)
            if kb.down(glfw.KEY_Q):
                offset_xyz[1] += 1
            elif kb.down(glfw.KEY_E):
                offset_xyz[1] -= 1
            elif kb.down(glfw.KEY_W):
                offset_xyz[2] += 1
            elif kb.down(glfw.KEY_S):
                offset_xyz[2] -= 1
            elif kb.down(glfw.KEY_A):
                offset_xyz[0] += 1
            elif kb.down(glfw.KEY_D):
                offset_xyz[0] -= 1

            offset_rpy *= 0.1
            offset_xyz *= 0.1
            self._free_rpy += offset_rpy
            self._free_xyz += offset_xyz

            if kb.down(glfw.KEY_B):
                self._free_rpy = np.zeros(3)
                self._free_xyz = np.zeros(3)

            quat = euler_to_quat(self._free_rpy)
            trans = mn.Matrix4.from_(quat.to_matrix(), mn.Vector3(*self._free_xyz))
            env._sim._sensors["third_rgb"]._sensor_object.node.transformation = trans
            step_result = env._sim.get_sensor_observations()
            return step_result

        return step_result


def play_env(env, args, config):
    render_steps_limit = None
    if args.no_render:
        render_steps_limit = DEFAULT_RENDER_STEPS_LIMIT

    use_arm_actions = None
    if args.load_actions is not None:
        with open(args.load_actions, "rb") as f:
            use_arm_actions = np.load(f)
            logger.info("Loaded arm actions")

    obs = env.reset()

    renderer: Optional[GLFWRenderer] = None
    kb: Optional[GLFWInput] = None
    draw_obs = None

    if not args.no_render:
        draw_obs = observations_to_image(obs, {})
        renderer = GLFWRenderer(draw_obs.shape[1], draw_obs.shape[0], "Habitat Interactive Play")
        kb = GLFWInput(renderer.window)

    update_idx = 0
    target_fps = 60.0
    prev_time = time.time()
    all_obs = []
    total_reward = 0
    all_arm_actions: List[float] = []
    agent_to_control = 0

    free_cam = FreeCamHelper()
    gfx_measure = env.task.measurements.measures.get(GfxReplayMeasure.cls_uuid, None)
    is_multi_agent = len(env._sim.agents_mgr) > 1

    humanoid_controller = None
    if args.use_humanoid_controller:
        humanoid_controller = HumanoidRearrangeController(args.walk_pose_path)
        humanoid_controller.reset(env._sim.articulated_agent.base_pos)

    while True:
        if args.save_actions and len(all_arm_actions) > args.save_actions_count:
            # quit when action recording queue is full
            break

        if render_steps_limit is not None and update_idx > render_steps_limit:
            break

        if not args.no_render:
            assert renderer is not None
            assert kb is not None

            if renderer.should_close():
                break
            kb.poll()

            if is_multi_agent and kb.pressed_once(glfw.KEY_X):
                agent_to_control = (agent_to_control + 1) % len(env._sim.agents_mgr)
                logger.info(f"Controlled agent changed. Controlling agent {agent_to_control}.")
        else:
            kb = None

        step_result, arm_action, end_ep = get_input_vel_ctlr(
            args.no_render,
            args.cfg,
            use_arm_actions[update_idx] if use_arm_actions is not None else None,
            env,
            not free_cam.is_free_cam_mode,
            agent_to_control,
            args.control_humanoid,
            humanoid_controller=humanoid_controller,
            kb=kb,
        )

        if not args.no_render and kb is not None and kb.down(glfw.KEY_C):
            pddl_action = env.task.actions["pddl_apply_action"]
            logger.info("Actions:")
            actions = pddl_action._action_ordering
            for i, action in enumerate(actions):
                logger.info(f"{i}: {action}")
            entities = pddl_action._entities_list
            logger.info("Entities")
            for i, entity in enumerate(entities):
                logger.info(f"{i}: {entity}")

            action_sel = input("Enter Action Selection: ")
            entity_sel = input("Enter Entity Selection: ")
            action_sel = int(action_sel)
            entity_sel = [int(x) + 1 for x in entity_sel.split(",")]

            ac = np.zeros(pddl_action.action_space["pddl_action"].shape[0])
            ac_start = pddl_action.get_pddl_action_start(action_sel)
            ac[ac_start: ac_start + len(entity_sel)] = entity_sel
            step_env(env, "pddl_apply_action", {"pddl_action": ac})

        if not args.no_render and kb is not None and kb.down(glfw.KEY_G):
            pred_list = env.task.sensor_suite.sensors["all_predicates"]._predicates_list
            pred_values = step_result["all_predicates"]
            logger.info("\nPredicate Truth Values:")
            for i, (pred, pred_value) in enumerate(zip(pred_list, pred_values)):
                logger.info(f"{i}: {pred.compact_str} = {pred_value}")

        if step_result is None:
            break

        if end_ep:
            total_reward = 0
            if gfx_measure is not None:
                gfx_measure.get_metric(force_get=True)
            env.reset()

        if not args.no_render and kb is not None:
            step_result = free_cam.update(env, step_result, kb)

        all_arm_actions.append(arm_action)
        update_idx += 1

        if use_arm_actions is not None and update_idx >= len(use_arm_actions):
            break

        obs = step_result
        info = env.get_metrics()

        reward_key = [k for k in info if "reward" in k]
        reward = info[reward_key[0]] if len(reward_key) > 0 else 0.0

        total_reward += reward
        info["Total Reward"] = total_reward

        if free_cam.is_free_cam_mode:
            assert draw_obs is not None
            cam = obs["third_rgb"]
            use_ob = np.zeros(draw_obs.shape, dtype=np.uint8)
            use_ob[:, : cam.shape[1]] = cam[:, :, :3]
        else:
            use_ob = observations_to_image(obs, info)
            if not args.skip_render_text:
                use_ob = overlay_frame(use_ob, info)

        draw_ob = np.array(use_ob, copy=True)

        if not args.no_render:
            assert renderer is not None
            renderer.render(draw_ob)

        if args.save_obs:
            all_obs.append(draw_ob)

        if env.episode_over:
            total_reward = 0
            env.reset()

        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0 / target_fps - diff, 0)
        time.sleep(delay)
        prev_time = curr_time

    if args.save_actions:
        if len(all_arm_actions) < args.save_actions_count:
            raise ValueError(
                f"Only did {len(all_arm_actions)} actions but {args.save_actions_count} are required"
            )
        all_arm_actions = all_arm_actions[: args.save_actions_count]
        os.makedirs(SAVE_ACTIONS_DIR, exist_ok=True)
        save_path = osp.join(SAVE_ACTIONS_DIR, args.save_actions_fname)
        with open(save_path, "wb") as f:
            np.save(f, all_arm_actions)
        logger.info(f"Saved actions to {save_path}")

        if renderer is not None:
            renderer.close()
        return

    if args.save_obs:
        all_obs = np.array(all_obs, dtype=np.uint8)
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        vut.make_video(
            np.expand_dims(all_obs, 1),
            0,
            "color",
            osp.join(SAVE_VIDEO_DIR, args.save_obs_fname),
        )

    if gfx_measure is not None:
        gfx_str = gfx_measure.get_metric(force_get=True)
        write_gfx_replay(gfx_str, config.habitat.task, env.current_episode.episode_id)

    if renderer is not None:
        renderer.close()


def has_glfw():
    return (glfw is not None) and (gl is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument("--save-actions-fname", type=str, default="play_actions.txt")
    parser.add_argument(
        "--save-actions-count",
        type=int,
        default=200,
        help="""
            The number of steps the saved action trajectory is clipped to. NOTE
            the episode must be at least this long or it will terminate with
            error.
            """,
    )
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument("--skip-render-text", action="store_true", default=False)
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control.",
    )

    parser.add_argument(
        "--control-humanoid",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--use-humanoid-controller",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument("--walk-pose-path", type=str, default=DEFAULT_POSE_PATH)

    args = parser.parse_args()

    if not args.no_render and not has_glfw():
        raise ImportError(
            "Need GLFW + PyOpenGL (run `pip install glfw PyOpenGL`)"
        )

    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        if not args.same_task:
            sim_config.debug_render = True
            agent_config = get_agent_config(sim_config=sim_config)
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    )
                }
            )
            if "pddl_success" in task_config.measurements:
                task_config.measurements.pddl_success.must_call_stop = False
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = False
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = -1.0

        if args.gfx:
            sim_config.habitat_sim_v0.enable_gfx_replay_save = True
            task_config.measurements.update(
                {"gfx_replay_measure": GfxReplayMeasureMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0

        if args.control_humanoid:
            args.disable_inverse_kinematics = True

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. "
                    "Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"

        if task_config.type == "RearrangePddlTask-v0":
            task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()

    with habitat.Env(config=config) as env:
        play_env(env, args, config)
