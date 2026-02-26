from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from telemetry_runtime import JsonlTelemetryLogger, now_perf
except Exception:  # pragma: no cover - optional telemetry dependency
    JsonlTelemetryLogger = None  # type: ignore

    def now_perf() -> float:
        return time.perf_counter()


DEFAULT_G1_START_Z = 0.43
DEFAULT_G1_GROUND_CLEARANCE_Z = 0.03
NAV_CMD_DEADBAND = 1e-4
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


def _default_usd_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "g1" / "g1_d455.usd"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [isaacsim_runner] %(message)s",
    )


def _add_flat_grid_environment(stage_obj) -> bool:
    flat_grid_targets = ("/World/Environment/FlatGrid", "/World/FlatGrid", "/FlatGrid")
    for target in flat_grid_targets:
        if stage_obj.GetPrimAtPath(target).IsValid():
            logging.info("Flat grid already present at %s", target)
            return False

    try:
        from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore
        from isaacsim.storage.native import get_assets_root_path  # type: ignore
    except Exception as exc:
        logging.warning("Could not import stage utilities for flat grid setup: %s", exc)
        return False

    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        logging.warning("Could not resolve Isaac Sim assets root. Skipping flat grid setup.")
        return False

    flat_grid_usd = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
    if stage_obj.GetPrimAtPath("/World/Environment").IsValid():
        target_prim = "/World/Environment/FlatGrid"
    elif stage_obj.GetPrimAtPath("/World").IsValid():
        target_prim = "/World/FlatGrid"
    else:
        target_prim = "/FlatGrid"
    try:
        add_reference_to_stage(flat_grid_usd, target_prim)
        logging.info("Added flat grid environment: usd=%s, prim=%s", flat_grid_usd, target_prim)
        return True
    except Exception as exc:
        logging.warning("Failed to add flat grid environment: %s", exc)
        return False


def _ensure_world_environment(stage_obj) -> None:
    try:
        from pxr import Usd, UsdGeom, UsdLux, UsdPhysics  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr modules for world setup: %s", exc)
        return

    def _ensure_xform(path: str) -> None:
        if not stage_obj.GetPrimAtPath(path).IsValid():
            UsdGeom.Xform.Define(stage_obj, path)
            logging.info("Created Xform prim: %s", path)

    _ensure_xform("/World")
    _ensure_xform("/World/Environment")
    _ensure_xform("/World/Robots")

    world_prim = stage_obj.GetPrimAtPath("/World")
    if world_prim.IsValid() and stage_obj.GetDefaultPrim() != world_prim:
        stage_obj.SetDefaultPrim(world_prim)
        logging.info("Set default prim: /World")

    physics_scene_path = "/World/PhysicsScene"
    if not stage_obj.GetPrimAtPath(physics_scene_path).IsValid():
        scene = UsdPhysics.Scene.Define(stage_obj, physics_scene_path)
        scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        logging.info("Created default physics scene: %s", physics_scene_path)

    light_path = "/World/Environment/KeyLight"
    if not stage_obj.GetPrimAtPath(light_path).IsValid():
        light = UsdLux.DistantLight.Define(stage_obj, light_path)
        light.CreateIntensityAttr(500.0)
        light.CreateAngleAttr(0.53)
        logging.info("Created default distant light: %s", light_path)

    def _move_prim(path_from: str, path_to: str) -> bool:
        src = stage_obj.GetPrimAtPath(path_from)
        if not src.IsValid():
            return False
        if stage_obj.GetPrimAtPath(path_to).IsValid():
            return True

        try:
            namespace_editor = Usd.NamespaceEditor(stage_obj)
            namespace_editor.MovePrimAtPath(path_from, path_to)
            if namespace_editor.ApplyEdits():
                logging.info("Moved prim: %s -> %s", path_from, path_to)
                return True
        except Exception as exc:
            logging.debug("Usd.NamespaceEditor move failed (%s -> %s): %s", path_from, path_to, exc)

        try:
            import omni.kit.commands  # type: ignore

            omni.kit.commands.execute(
                "MovePrim",
                path_from=path_from,
                path_to=path_to,
                keep_world_transform=True,
            )
            logging.info("Moved prim with MovePrim command: %s -> %s", path_from, path_to)
            return True
        except Exception as exc:
            logging.warning("Failed to move prim (%s -> %s): %s", path_from, path_to, exc)
            return False

    _move_prim("/FlatGrid", "/World/Environment/FlatGrid")
    _move_prim("/World/FlatGrid", "/World/Environment/FlatGrid")

    # Keep robot roots at their original authored paths.
    # Moving articulated roots with NamespaceEditor/MovePrim can leave stale
    # relationship targets on composed assets (especially bridge USDs), which
    # breaks physics stability.


def _prepare_internal_ros2_environment() -> None:
    isaac_root = Path(os.environ.get("ISAAC_SIM_ROOT", r"C:\isaac-sim"))
    ros2_lib = isaac_root / "exts" / "isaacsim.ros2.bridge" / "humble" / "lib"
    if not ros2_lib.exists():
        return

    os.environ.setdefault("ROS_DISTRO", "humble")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    path_sep = os.pathsep
    current_path = os.environ.get("PATH", "")
    entries = current_path.split(path_sep) if current_path else []
    ros2_lib_str = str(ros2_lib)
    if ros2_lib_str not in entries:
        os.environ["PATH"] = f"{current_path}{path_sep}{ros2_lib_str}" if current_path else ros2_lib_str


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
        return {"matched_dof": matched, "num_dof": num_dof, "groups": [], "remaining_dof_count": max(0, num_dof - matched)}

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


def _quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _read_base_pose(stage_obj, robot_prim_path: str) -> Optional[Dict[str, float]]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception:
        return None

    prim = stage_obj.GetPrimAtPath(robot_prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
        return None
    try:
        world_t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        trans = world_t.ExtractTranslation()
        quat = world_t.ExtractRotationQuat()
        imag = quat.GetImaginary()
        qx = float(imag[0])
        qy = float(imag[1])
        qz = float(imag[2])
        qw = float(quat.GetReal())
        roll, pitch, yaw = _quat_to_rpy(qw, qx, qy, qz)
        return {
            "x": float(trans[0]),
            "y": float(trans[1]),
            "height": float(trans[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        }
    except Exception:
        return None


def _get_translate_op(xform) -> Optional[object]:
    try:
        from pxr import UsdGeom  # type: ignore
    except Exception:
        return None

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return None


def _count_rigid_bodies_under(stage_obj, prim, usd_physics) -> int:
    if not prim.IsValid():
        return 0
    prefix = prim.GetPath()
    count = 0
    for p in stage_obj.Traverse():
        if not p.GetPath().HasPrefix(prefix):
            continue
        if p.HasAPI(usd_physics.RigidBodyAPI):
            count += 1
    return count


def _resolve_robot_placement_prim(stage_obj, robot_prim_path: str):
    prim = stage_obj.GetPrimAtPath(robot_prim_path)
    if not prim.IsValid():
        return prim

    try:
        from pxr import UsdGeom, UsdPhysics  # type: ignore
    except Exception:
        # Fallback: If the articulation root path points to a link (e.g. pelvis), move the parent model prim.
        if not prim.GetAttribute("isaac:physics:robotJoints").IsValid():
            parent = prim.GetParent()
            if parent.IsValid() and parent.GetPath().pathString != "/":
                return parent
        return prim

    # Prefer a xformable ancestor that actually owns the articulated rigid bodies.
    # Some USDs report articulation root on a joint prim (e.g. ".../root_joint"), which has no rigid bodies.
    if prim.IsA(UsdGeom.Xformable):
        self_count = _count_rigid_bodies_under(stage_obj, prim, UsdPhysics)
        if self_count >= 5:
            return prim

    parent = prim.GetParent()
    while parent.IsValid() and parent.GetPath().pathString not in ("", "/"):
        if parent.IsA(UsdGeom.Xformable):
            parent_count = _count_rigid_bodies_under(stage_obj, parent, UsdPhysics)
            if parent_count >= 5:
                return parent
        parent = parent.GetParent()

    return prim


def _rebase_world_anchor_articulation_root(stage_obj, robot_prim_path: str) -> str:
    try:
        from pxr import PhysxSchema, Sdf, UsdPhysics  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr modules to inspect/rebase articulation root: %s", exc)
        return robot_prim_path

    root_prim = stage_obj.GetPrimAtPath(robot_prim_path)
    if not root_prim.IsValid() or root_prim.GetTypeName() != "PhysicsFixedJoint":
        return robot_prim_path
    if not root_prim.IsA(UsdPhysics.Joint):
        return robot_prim_path

    body0_rel = root_prim.GetRelationship("physics:body0")
    body1_rel = root_prim.GetRelationship("physics:body1")
    body0_targets = [str(path) for path in body0_rel.GetTargets()] if body0_rel and body0_rel.IsValid() else []
    body1_targets = [str(path) for path in body1_rel.GetTargets()] if body1_rel and body1_rel.IsValid() else []
    world_anchor = (len(body0_targets) == 0) or all(target in ("/World", "/world") for target in body0_targets)
    if (not world_anchor) or (not body1_targets):
        return robot_prim_path

    placement_prim = _resolve_robot_placement_prim(stage_obj, robot_prim_path)
    if not placement_prim.IsValid():
        return robot_prim_path

    enabled_attr = root_prim.GetAttribute("physics:jointEnabled")
    if not enabled_attr.IsValid():
        enabled_attr = root_prim.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool)
    enabled_attr.Set(False)

    try:
        root_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    except Exception:
        pass
    try:
        root_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    except Exception:
        pass

    if not placement_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(placement_prim)
    if not placement_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(placement_prim)

    rebased_path = placement_prim.GetPath().pathString
    logging.info(
        "Rebased world-anchor articulation root for floating base: old=%s new=%s body1=%s",
        robot_prim_path,
        rebased_path,
        body1_targets,
    )
    return rebased_path


def _ensure_robot_dynamic_flags(stage_obj, robot_prim_path: str) -> None:
    try:
        from pxr import UsdPhysics  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr UsdPhysics to normalize dynamic flags: %s", exc)
        return

    placement_prim = _resolve_robot_placement_prim(stage_obj, robot_prim_path)
    if not placement_prim.IsValid():
        return

    root_path = placement_prim.GetPath()
    rigid_bodies = 0
    kinematic_fixed = 0
    gravity_fixed = 0

    for prim in stage_obj.Traverse():
        if not prim.GetPath().HasPrefix(root_path):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_bodies += 1

        for attr_name in ("physics:kinematicEnabled", "physxRigidBody:kinematicEnabled"):
            attr = prim.GetAttribute(attr_name)
            if not attr.IsValid():
                continue
            try:
                if bool(attr.Get()):
                    attr.Set(False)
                    kinematic_fixed += 1
            except Exception:
                pass

        for attr_name in ("physxRigidBody:disableGravity", "physics:disableGravity"):
            attr = prim.GetAttribute(attr_name)
            if not attr.IsValid():
                continue
            try:
                if bool(attr.Get()):
                    attr.Set(False)
                    gravity_fixed += 1
            except Exception:
                pass

    if rigid_bodies > 0 and (kinematic_fixed > 0 or gravity_fixed > 0):
        logging.info(
            "Normalized rigid body flags under %s: rigid_bodies=%d kinematic_disabled=%d gravity_enabled=%d",
            placement_prim.GetPath().pathString,
            rigid_bodies,
            kinematic_fixed,
            gravity_fixed,
        )


def _find_motion_root_prim(stage_obj, robot_prim_path: str) -> str:
    if not robot_prim_path:
        return robot_prim_path
    prim = _resolve_robot_placement_prim(stage_obj, robot_prim_path)
    if prim.IsValid():
        return prim.GetPath().pathString
    return robot_prim_path


class NavigateCommandBridge:
    """Simple twist subscriber that applies base XY/yaw motion to the robot root."""

    def __init__(self, namespace: str, robot_root_prim_path: str, command_timeout_s: float = 0.6) -> None:
        self.namespace = namespace.strip("/")
        self.robot_root_prim_path = robot_root_prim_path
        self.command_timeout_s = float(command_timeout_s)

        self.available = False
        self._rclpy = None
        self._node = None
        self._sub = None
        self._last_cmd_ts = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._last_warn_ts = 0.0
        self._last_rx_log_ts = 0.0

    def start(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Twist
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("Navigate command bridge disabled (missing ROS2 libs): %s", exc)
            return

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("g1_navigate_command_bridge")
        topic = f"/{self.namespace}/cmd/navigate" if self.namespace else "/cmd/navigate"
        self._sub = self._node.create_subscription(Twist, topic, self._on_twist, 20)
        self.available = True
        logging.info(
            "Navigate command bridge ready: topic=%s target_prim=%s",
            topic,
            self.robot_root_prim_path,
        )

    def _on_twist(self, msg) -> None:
        self._vx = float(msg.linear.x)
        self._vy = float(msg.linear.y)
        self._wz = float(msg.angular.z)
        self._last_cmd_ts = time.time()
        if (self._last_cmd_ts - self._last_rx_log_ts) > 0.7:
            logging.info(
                "Navigate command rx: vx=%.3f vy=%.3f wz=%.3f",
                self._vx,
                self._vy,
                self._wz,
            )
            self._last_rx_log_ts = self._last_cmd_ts

    def spin_once(self) -> None:
        if self.available and self._node is not None and self._rclpy is not None:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def apply(self, stage_obj, dt: float) -> None:
        if not self.available:
            return
        if (time.time() - self._last_cmd_ts) > self.command_timeout_s:
            return
        if dt <= 0.0:
            return
        # Avoid re-authoring transform for effectively zero commands;
        # otherwise physics settling/gravity can be unintentionally suppressed.
        if (abs(self._vx) + abs(self._vy) + abs(self._wz)) <= NAV_CMD_DEADBAND:
            return

        try:
            from pxr import Gf, UsdGeom  # type: ignore
        except Exception:
            return

        prim = stage_obj.GetPrimAtPath(self.robot_root_prim_path)
        if not prim.IsValid():
            if time.time() - self._last_warn_ts > 2.0:
                logging.warning("Navigate command target prim missing: %s", self.robot_root_prim_path)
                self._last_warn_ts = time.time()
            return

        xform = UsdGeom.Xformable(prim)
        translate_op = _get_translate_op(xform)
        rotate_z_op = None
        for op in xform.GetOrderedXformOps():
            if rotate_z_op is None and op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                rotate_z_op = op

        if translate_op is None:
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(0.0, 0.0, DEFAULT_G1_START_Z))
        if rotate_z_op is None:
            rotate_z_op = xform.AddRotateZOp()
            rotate_z_op.Set(0.0)

        pos = translate_op.Get() or Gf.Vec3d(0.0, 0.0, DEFAULT_G1_START_Z)
        yaw_deg = float(rotate_z_op.Get() or 0.0)
        yaw = math.radians(yaw_deg)
        world_vx = self._vx * math.cos(yaw) - self._vy * math.sin(yaw)
        world_vy = self._vx * math.sin(yaw) + self._vy * math.cos(yaw)

        next_pos = Gf.Vec3d(
            float(pos[0]) + float(world_vx * dt),
            float(pos[1]) + float(world_vy * dt),
            float(pos[2]),
        )
        next_yaw = float(yaw_deg + math.degrees(self._wz * dt))
        translate_op.Set(next_pos)
        rotate_z_op.Set(next_yaw)

    def stop(self) -> None:
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        self._node = None
        self._sub = None


class MockRos2Publisher:
    """Optional ROS2 mock publisher for camera/depth/tf/joint_states/clock."""

    def __init__(self, namespace: str, publish_imu: bool, publish_compressed_color: bool) -> None:
        self.namespace = namespace.strip("/")
        self.publish_imu = publish_imu
        self.publish_compressed_color = publish_compressed_color
        self.available = False
        self.node = None
        self.executor = None
        self._imports = {}
        self._last_log = 0.0

        try:
            import rclpy
            from builtin_interfaces.msg import Time
            from geometry_msgs.msg import TransformStamped
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rosgraph_msgs.msg import Clock
            from sensor_msgs.msg import CompressedImage, Image, Imu, JointState
            from tf2_msgs.msg import TFMessage

            self._imports = {
                "rclpy": rclpy,
                "Time": Time,
                "TransformStamped": TransformStamped,
                "SingleThreadedExecutor": SingleThreadedExecutor,
                "Node": Node,
                "Clock": Clock,
                "Image": Image,
                "CompressedImage": CompressedImage,
                "Imu": Imu,
                "JointState": JointState,
                "TFMessage": TFMessage,
            }
            self.available = True
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("ROS2 python packages not found. Mock ROS2 publish disabled: %s", exc)

    def start(self) -> None:
        if not self.available:
            return
        rclpy = self._imports["rclpy"]
        Node = self._imports["Node"]
        Executor = self._imports["SingleThreadedExecutor"]
        Clock = self._imports["Clock"]
        Image = self._imports["Image"]
        JointState = self._imports["JointState"]
        TFMessage = self._imports["TFMessage"]
        TransformStamped = self._imports["TransformStamped"]
        Imu = self._imports["Imu"]
        CompressedImage = self._imports["CompressedImage"]

        rclpy.init(args=None)
        self.node = Node("isaacsim_runner_mock_pub")
        ns = f"/{self.namespace}" if self.namespace else ""

        self._color_pub = self.node.create_publisher(Image, f"{ns}/camera/color/image_raw", 10)
        self._depth_pub = self.node.create_publisher(Image, f"{ns}/camera/depth/image_raw", 10)
        self._joint_pub = self.node.create_publisher(JointState, f"{ns}/joint_states", 10)
        self._clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        self._tf_pub = self.node.create_publisher(TFMessage, "/tf", 10)
        self._color_compressed_pub = None
        self._imu_pub = None
        if self.publish_compressed_color:
            self._color_compressed_pub = self.node.create_publisher(
                CompressedImage, f"{ns}/camera/color/image_raw/compressed", 10
            )
        if self.publish_imu:
            self._imu_pub = self.node.create_publisher(Imu, f"{ns}/imu", 10)

        self._msg_clock = Clock()
        self._msg_color = Image()
        self._msg_color.height = 480
        self._msg_color.width = 640
        self._msg_color.encoding = "rgb8"
        self._msg_color.step = self._msg_color.width * 3
        self._msg_color.data = bytes(self._msg_color.height * self._msg_color.step)
        self._msg_color.header.frame_id = f"{self.namespace}/camera_color_optical_frame"

        self._msg_depth = Image()
        self._msg_depth.height = 480
        self._msg_depth.width = 640
        self._msg_depth.encoding = "16UC1"
        self._msg_depth.step = self._msg_depth.width * 2
        self._msg_depth.data = bytes(self._msg_depth.height * self._msg_depth.step)
        self._msg_depth.header.frame_id = f"{self.namespace}/camera_depth_optical_frame"

        self._msg_joint = JointState()
        self._msg_joint.name = ["joint_1", "joint_2", "joint_3"]
        self._msg_joint.position = [0.0, 0.0, 0.0]

        self._msg_tf = TFMessage()
        tf = TransformStamped()
        tf.header.frame_id = "map"
        tf.child_frame_id = f"{self.namespace}/base_link"
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = DEFAULT_G1_START_Z
        tf.transform.rotation.w = 1.0
        self._msg_tf.transforms = [tf]

        self._msg_imu = Imu()
        self._msg_imu.header.frame_id = f"{self.namespace}/imu_link"
        self._msg_imu.orientation.w = 1.0
        self._msg_color_compressed = CompressedImage()
        self._msg_color_compressed.header.frame_id = f"{self.namespace}/camera_color_optical_frame"
        self._msg_color_compressed.format = "jpeg"
        self._msg_color_compressed.data = b""

        self.executor = Executor()
        self.executor.add_node(self.node)
        logging.info("Mock ROS2 publishers started on namespace '/%s'", self.namespace)

    def publish_once(self) -> None:
        if not self.available or self.node is None:
            return

        now = self.node.get_clock().now().to_msg()
        self._msg_color.header.stamp = now
        self._msg_depth.header.stamp = now
        self._msg_joint.header.stamp = now
        self._msg_clock.clock = now
        self._msg_tf.transforms[0].header.stamp = now
        self._msg_imu.header.stamp = now
        self._msg_color_compressed.header.stamp = now

        self._color_pub.publish(self._msg_color)
        self._depth_pub.publish(self._msg_depth)
        if self._color_compressed_pub is not None:
            self._color_compressed_pub.publish(self._msg_color_compressed)
        self._joint_pub.publish(self._msg_joint)
        self._clock_pub.publish(self._msg_clock)
        self._tf_pub.publish(self._msg_tf)
        if self._imu_pub is not None:
            self._imu_pub.publish(self._msg_imu)

        if time.time() - self._last_log >= 5.0:
            logging.info(
                "Publishing mock topics: /%s/camera/{color,depth}/image_raw, /%s/joint_states, /tf, /clock%s",
                self.namespace,
                self.namespace,
                (
                    f", /{self.namespace}/camera/color/image_raw/compressed"
                    if self.publish_compressed_color
                    else ""
                )
                + (f", /{self.namespace}/imu" if self.publish_imu else ""),
            )
            self._last_log = time.time()

    def spin_once(self) -> None:
        if self.executor is not None:
            self.executor.spin_once(timeout_sec=0.0)

    def stop(self) -> None:
        if not self.available:
            return
        rclpy = self._imports["rclpy"]
        if self.executor is not None and self.node is not None:
            self.executor.remove_node(self.node)
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _run_mock_loop(args: argparse.Namespace) -> None:
    ros2 = MockRos2Publisher(args.namespace, args.publish_imu, args.publish_compressed_color)
    ros2.start()

    logging.info("Running Isaac Sim mock loop with USD: %s", args.usd)
    logging.info("USD transform edits are intentionally disabled.")

    try:
        sleep_s = max(0.01, 1.0 / float(args.rate_hz))
        last_heartbeat = 0.0
        while True:
            ros2.publish_once()
            ros2.spin_once()
            if not ros2.available and (time.time() - last_heartbeat) >= 5.0:
                logging.info(
                    "Mock heartbeat: would publish RGB/Depth/TF/JointState using D455 in g1_d455.usd."
                )
                last_heartbeat = time.time()
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        logging.info("Mock runner interrupted by user.")
    finally:
        ros2.stop()


def _find_robot_prim_path(stage_obj) -> Optional[str]:
    try:
        from pxr import UsdPhysics
    except Exception:
        return None

    articulation_roots = []
    for prim in stage_obj.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim.GetPath().pathString)
    if articulation_roots:
        # Prefer the shortest path (closest to stage root) if there are multiple roots.
        articulation_roots.sort(key=lambda p: (p.count("/"), len(p)))
        selected = articulation_roots[0]
        logging.info("Detected articulation root prim(s): %s -> selected=%s", articulation_roots, selected)
        return selected

    preferred = (
        "/World/Robots/g1_29dof_with_hand_rev_1_0",
        "/g1_29dof_with_hand_rev_1_0",
        "/World/g1_29dof_with_hand_rev_1_0",
        "/World/Robots/g1",
        "/g1",
        "/World/g1",
    )
    for path in preferred:
        prim = stage_obj.GetPrimAtPath(path)
        if prim.IsValid():
            logging.info(
                "Falling back to preferred robot prim path without ArticulationRootAPI: %s",
                path,
            )
            return path

    for prim in stage_obj.Traverse():
        if prim.GetAttribute("isaac:physics:robotJoints").IsValid():
            path = prim.GetPath().pathString
            logging.info(
                "Falling back to robotJoints attribute prim path without ArticulationRootAPI: %s",
                path,
            )
            return path

    return None


def _set_robot_start_height(stage_obj, robot_prim_path: str, z: float) -> None:
    try:
        from pxr import Gf, Usd, UsdGeom  # type: ignore
    except Exception as exc:
        logging.warning("Could not import pxr modules to set robot start height: %s", exc)
        return

    prim = _resolve_robot_placement_prim(stage_obj, robot_prim_path)
    if not prim.IsValid():
        logging.warning("Robot prim is not valid; skip start height set: %s", robot_prim_path)
        return

    if not prim.IsA(UsdGeom.Xformable):
        logging.warning("Robot prim is not xformable; skip start height set: %s", robot_prim_path)
        return

    xform = UsdGeom.Xformable(prim)
    translate_op = _get_translate_op(xform)
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    current = translate_op.Get() or Gf.Vec3d(0.0, 0.0, z)
    try:
        x_pos = float(current[0])
        y_pos = float(current[1])
        z_pos = float(current[2])
    except Exception:
        x_pos = 0.0
        y_pos = 0.0
        z_pos = z

    aligned = None
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        )
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    except Exception as exc:
        logging.warning("Failed to compute robot world bounds for spawn placement: %s", exc)

    if aligned is None or aligned.IsEmpty():
        translate_op.Set((x_pos, y_pos, z))
        logging.info(
            "Set robot start height via fallback: prim=%s z=%.3f (bounds unavailable)",
            prim.GetPath().pathString,
            z,
        )
        return

    min_z = float(aligned.GetMin()[2])
    lift_z = max(0.0, DEFAULT_G1_GROUND_CLEARANCE_Z - min_z)
    target_z = z_pos + lift_z
    translate_op.Set((x_pos, y_pos, target_z))
    logging.info(
        "Adjusted robot spawn clearance: prim=%s min_z=%.3f clearance=%.3f lift=%.3f final_z=%.3f",
        prim.GetPath().pathString,
        min_z,
        DEFAULT_G1_GROUND_CLEARANCE_Z,
        lift_z,
        target_z,
    )


def _find_camera_prim_path(stage_obj) -> Optional[str]:
    camera_paths = []
    for prim in stage_obj.Traverse():
        if prim.GetTypeName() == "Camera":
            camera_paths.append(prim.GetPath().pathString)
    if not camera_paths:
        return None

    # Prefer physical sensor camera prims, especially color camera.
    for path in camera_paths:
        lowered = path.lower()
        if "color" in lowered and "omniversekit_" not in lowered:
            return path
    for path in camera_paths:
        lowered = path.lower()
        if ("d435" in lowered or "rsd455" in lowered or "camera" in lowered) and "omniversekit_" not in lowered:
            return path
    return camera_paths[0]


def _setup_ros2_joint_and_tf_graph(namespace: str, robot_prim_path: str) -> None:
    import omni.graph.core as og  # type: ignore
    import usdrt.Sdf  # type: ignore

    graph_path = "/G1ROS2Bridge"
    cmd_topic = f"/{namespace}/cmd/joint_commands"
    joint_state_topic = f"/{namespace}/joint_states"

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishTF.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", "/clock"),
                ("PublishJointState.inputs:topicName", joint_state_topic),
                ("SubscribeJointState.inputs:topicName", cmd_topic),
                ("PublishTF.inputs:topicName", "/tf"),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(robot_prim_path)]),
                ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(robot_prim_path)]),
                ("ArticulationController.inputs:robotPath", robot_prim_path),
            ],
        },
    )

    logging.info(
        "ROS2 bridge graph ready: robot=%s, cmd_topic=%s, joint_state_topic=%s",
        robot_prim_path,
        cmd_topic,
        joint_state_topic,
    )


def _setup_ros2_camera_graph(namespace: str, camera_prim_path: str) -> None:
    import omni.graph.core as og  # type: ignore
    import usdrt.Sdf  # type: ignore

    graph_path = "/G1ROSCamera"
    color_topic = f"/{namespace}/camera/color/image_raw"
    depth_topic = f"/{namespace}/camera/depth/image_raw"
    camera_info_topic = f"/{namespace}/camera/color/camera_info"

    keys = og.Controller.Keys
    ros_camera_graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("CameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperRgb.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperInfo.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperDepth.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperRgb.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperInfo.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperDepth.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                # Keep GUI viewport interactive by using a dedicated offscreen render product.
                ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_prim_path)]),
                ("CreateRenderProduct.inputs:width", 640),
                ("CreateRenderProduct.inputs:height", 480),
                ("CameraHelperRgb.inputs:frameId", f"{namespace}/camera_color_optical_frame"),
                ("CameraHelperRgb.inputs:topicName", color_topic),
                ("CameraHelperRgb.inputs:type", "rgb"),
                ("CameraHelperInfo.inputs:frameId", f"{namespace}/camera_color_optical_frame"),
                ("CameraHelperInfo.inputs:topicName", camera_info_topic),
                ("CameraHelperDepth.inputs:frameId", f"{namespace}/camera_depth_optical_frame"),
                ("CameraHelperDepth.inputs:topicName", depth_topic),
                ("CameraHelperDepth.inputs:type", "depth"),
            ],
        },
    )
    og.Controller.evaluate_sync(ros_camera_graph)
    logging.info(
        "ROS2 camera graph ready: camera=%s, topics=[%s, %s, %s]",
        camera_prim_path,
        color_topic,
        depth_topic,
        camera_info_topic,
    )


def _run_native_isaac(args: argparse.Namespace) -> None:
    _prepare_internal_ros2_environment()
    navigate_bridge: Optional[NavigateCommandBridge] = None
    telemetry = None
    if JsonlTelemetryLogger is not None:
        try:
            telemetry = JsonlTelemetryLogger(
                phase=os.environ.get("AURA_TELEMETRY_PHASE", "standing"),
                component="isaac_runner",
            )
        except Exception as exc:
            logging.warning("Failed to initialize runtime telemetry logger: %s", exc)

    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception as exc:
        try:
            from omni.isaac.kit import SimulationApp  # type: ignore
        except Exception:
            logging.warning("Isaac Sim modules unavailable. Falling back to mock mode: %s", exc)
            _run_mock_loop(args)
            return

    headless = not bool(args.gui)
    sim_app_config = {"headless": headless}
    if args.gui:
        # Use editor-style rendering defaults when GUI is requested.
        sim_app_config.update(
            {
                "hide_ui": False,
                "display_options": 3286,
                "window_width": 1920,
                "window_height": 1080,
            }
        )
    isaac_root = Path(os.environ.get("ISAAC_SIM_ROOT", r"C:\isaac-sim"))
    if args.gui:
        experience = isaac_root / "apps" / "isaacsim.exp.full.kit"
    else:
        experience = isaac_root / "apps" / "isaacsim.exp.base.python.kit"

    simulation_app = (
        SimulationApp(sim_app_config, experience=str(experience))
        if experience.exists()
        else SimulationApp(sim_app_config)
    )
    try:
        import omni.usd  # type: ignore
        from isaacsim.core.api import SimulationContext  # type: ignore
        from omni.isaac.core.articulations import Articulation  # type: ignore
        from isaacsim.core.utils.extensions import enable_extension  # type: ignore

        usd_path = Path(args.usd).resolve()
        if not usd_path.exists():
            raise FileNotFoundError(f"USD file not found: {usd_path}")

        enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()

        context = omni.usd.get_context()
        context.open_stage(str(usd_path))
        for _ in range(90):
            simulation_app.update()

        stage_obj = context.get_stage()
        _ensure_world_environment(stage_obj)
        for _ in range(15):
            simulation_app.update()
        stage_obj = context.get_stage()

        if _add_flat_grid_environment(stage_obj):
            for _ in range(60):
                simulation_app.update()
            stage_obj = context.get_stage()
        _ensure_world_environment(stage_obj)
        for _ in range(15):
            simulation_app.update()
        stage_obj = context.get_stage()

        robot_prim_path = _find_robot_prim_path(stage_obj)
        motion_root_prim: Optional[str] = None
        camera_prim_path = _find_camera_prim_path(stage_obj)
        if robot_prim_path:
            robot_prim_path = _rebase_world_anchor_articulation_root(stage_obj, robot_prim_path)
            _ensure_robot_dynamic_flags(stage_obj, robot_prim_path)
            _set_robot_start_height(stage_obj, robot_prim_path, DEFAULT_G1_START_Z)
            motion_root_prim = _find_motion_root_prim(stage_obj, robot_prim_path)
            if args.enable_navigate_bridge:
                navigate_bridge = NavigateCommandBridge(args.namespace, motion_root_prim)
                navigate_bridge.start()
            else:
                logging.info(
                    "Navigate command bridge disabled by default (physics-safe). "
                    "Use --enable-navigate-bridge to force-enable root transform control."
                )

        logging.info("Loaded USD stage: %s", usd_path)
        logging.info("Isaac Sim display mode: %s", "GUI" if args.gui else "headless")
        if robot_prim_path:
            logging.info("Resolved articulation root path: %s", robot_prim_path)
        if telemetry is not None:
            telemetry.log(
                {
                    "event": "runner_startup",
                    "usd_path": str(usd_path),
                    "articulation_root_path": str(robot_prim_path or ""),
                    "motion_root_prim_path": str(motion_root_prim or ""),
                    "requested_rate_hz": float(args.rate_hz),
                }
            )
        if robot_prim_path:
            _setup_ros2_joint_and_tf_graph(args.namespace, robot_prim_path)
        else:
            logging.warning("Could not detect articulation robot prim. Joint command bridge is disabled.")
        enable_camera_bridge = bool(camera_prim_path) and (
            (not args.gui) or bool(args.enable_camera_bridge_in_gui)
        )
        if enable_camera_bridge:
            _setup_ros2_camera_graph(args.namespace, camera_prim_path)
        elif camera_prim_path and args.gui:
            logging.info(
                "GUI mode: ROS camera bridge is disabled by default to keep viewport interactive. "
                "Use --enable-camera-bridge-in-gui to force-enable it."
            )
        else:
            logging.warning("Could not find a Camera prim in the stage. RGB/Depth ROS topics are disabled.")

        if abs(float(args.rate_hz) - 50.0) > 1e-6:
            logging.warning("Overriding --rate-hz=%.3f to fixed 50Hz for control-loop stability telemetry.", float(args.rate_hz))
        control_hz = 50.0
        simulation_context = SimulationContext(
            physics_dt=1.0 / 200.0,
            rendering_dt=1.0 / control_hz,
            stage_units_in_meters=1.0,
        )
        cmd_dt = 1.0 / 50.0
        if telemetry is not None:
            telemetry.log(
                {
                    "event": "runner_timing_config",
                    "physics_dt": 1.0 / 200.0,
                    "control_dt": cmd_dt,
                    "render_dt": cmd_dt,
                }
            )
        simulation_context.initialize_physics()
        simulation_context.play()
        simulation_context.step(render=False)
        dof_snapshot: Dict[str, Any] = {}
        pd_summary: Dict[str, Any] = {}
        if robot_prim_path:
            try:
                robot = Articulation(robot_prim_path)
                robot.initialize()
                dof_snapshot = _log_robot_dof_snapshot(robot)
                pd_summary = _apply_g1_pd_gains(robot)
                if telemetry is not None:
                    telemetry.log(
                        {
                            "event": "dof_snapshot",
                            "articulation_root_path": str(robot_prim_path),
                            "num_dof": dof_snapshot.get("num_dof"),
                            "dof_names": dof_snapshot.get("dof_names"),
                        }
                    )
                    telemetry.log(
                        {
                            "event": "pd_gains_applied",
                            "matched_dof": pd_summary.get("matched_dof"),
                            "num_dof": pd_summary.get("num_dof"),
                            "remaining_dof_count": pd_summary.get("remaining_dof_count"),
                            "pd_groups": pd_summary.get("groups", []),
                        }
                    )
            except Exception as exc:
                logging.warning("Failed to initialize articulation for gain setup: %s", exc)
        loop_step_idx = 0
        loop_prev_start: Optional[float] = None
        loop_dt_window: list[float] = []
        base_prev_pose: Optional[Dict[str, float]] = None
        base_pose_path = motion_root_prim if motion_root_prim else robot_prim_path
        wall_start = now_perf()
        while simulation_app.is_running():
            t_loop_start = now_perf()
            if loop_prev_start is None:
                loop_dt = cmd_dt
            else:
                loop_dt = max(0.0, t_loop_start - loop_prev_start)
            loop_prev_start = t_loop_start
            if navigate_bridge is not None:
                navigate_bridge.spin_once()
                navigate_bridge.apply(stage_obj, cmd_dt)
            simulation_context.step(render=True)
            t_loop_end = now_perf()
            elapsed = max(0.0, t_loop_end - t_loop_start)
            overrun_ms = max(0.0, (elapsed - cmd_dt) * 1000.0)
            loop_dt_window.append(loop_dt)
            if len(loop_dt_window) > 400:
                loop_dt_window = loop_dt_window[-400:]
            publish_hz_window = None
            if loop_dt_window:
                avg_loop_dt = sum(loop_dt_window[-50:]) / len(loop_dt_window[-50:])
                if avg_loop_dt > 0.0:
                    publish_hz_window = 1.0 / avg_loop_dt
            loop_step_idx += 1
            rtf = None
            wall_elapsed = t_loop_end - wall_start
            if wall_elapsed > 1e-6:
                rtf = (loop_step_idx * cmd_dt) / wall_elapsed
            base_pose = _read_base_pose(stage_obj, base_pose_path) if base_pose_path else None
            base_roll = None
            base_pitch = None
            base_yaw = None
            base_height = None
            fall_flag = False
            slip_flag = False
            if base_pose is not None:
                base_roll = float(base_pose["roll"])
                base_pitch = float(base_pose["pitch"])
                base_yaw = float(base_pose["yaw"])
                base_height = float(base_pose["height"])
                fall_flag = bool(abs(base_roll) > 0.9 or abs(base_pitch) > 0.9 or base_height < 0.22)
                if base_prev_pose is not None and loop_dt > 1e-6:
                    dx = float(base_pose["x"] - base_prev_pose["x"])
                    dy = float(base_pose["y"] - base_prev_pose["y"])
                    base_speed_xy = math.sqrt(dx * dx + dy * dy) / loop_dt
                    cmd_speed = 0.0
                    if navigate_bridge is not None:
                        cmd_speed = math.sqrt(float(navigate_bridge._vx) ** 2 + float(navigate_bridge._vy) ** 2)
                    slip_flag = bool((cmd_speed < 0.05 and base_speed_xy > 0.20) or (cmd_speed > 0.20 and base_speed_xy < 0.01))
                base_prev_pose = base_pose
            if telemetry is not None:
                rec: Dict[str, Any] = {
                    "event": "runner_loop",
                    "step_idx": loop_step_idx,
                    "loop_dt": float(loop_dt),
                    "loop_overrun_ms": float(overrun_ms),
                    "publish_hz_window": publish_hz_window,
                    "rtf": rtf,
                    "fall_flag": bool(fall_flag),
                    "slip_flag": bool(slip_flag),
                }
                if base_roll is not None:
                    rec["base_roll"] = base_roll
                if base_pitch is not None:
                    rec["base_pitch"] = base_pitch
                if base_yaw is not None:
                    rec["base_yaw"] = base_yaw
                if base_height is not None:
                    rec["base_height"] = base_height
                telemetry.log(rec)
        simulation_context.stop()
    except KeyboardInterrupt:
        logging.info("Isaac Sim runner interrupted by user.")
    finally:
        if navigate_bridge is not None:
            navigate_bridge.stop()
        if telemetry is not None:
            try:
                telemetry.flush()
                telemetry.close()
            except Exception:
                pass
        simulation_app.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isaac Sim 4.2 runner for g1_d455.usd on Windows."
    )
    parser.add_argument("--usd", type=str, default=str(_default_usd_path()))
    parser.add_argument("--namespace", type=str, default="g1")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--publish-imu", action="store_true")
    parser.add_argument("--publish-compressed-color", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Run native Isaac Sim with GUI window.")
    parser.add_argument(
        "--enable-camera-bridge-in-gui",
        action="store_true",
        help="Enable ROS camera graph in GUI mode (may impact viewport interactivity on some systems).",
    )
    parser.add_argument(
        "--enable-navigate-bridge",
        action="store_true",
        help="Enable /<namespace>/cmd/navigate root transform bridge (disabled by default for physics safety).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode even if Isaac Sim python modules are available.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    usd_path = Path(args.usd).resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")
    logging.info("Using USD: %s", usd_path)

    if args.mock:
        _run_mock_loop(args)
        return
    _run_native_isaac(args)


if __name__ == "__main__":
    main()
