from __future__ import annotations

import argparse
import ast
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.isaacsim_runner.config.base import DEFAULT_G1_START_Z
from apps.isaacsim_runner.stage.prims import get_translate_op, normalize_prim_path
from apps.isaacsim_runner.stage.robot import find_robot_prim_path


@dataclass
class OccupancyMapSpec:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    grid: list[list[int]]  # [my][mx], where (0,0) is map lower-left

    def in_bounds(self, mx: int, my: int) -> bool:
        return 0 <= mx < self.width and 0 <= my < self.height

    def is_occupied(self, mx: int, my: int) -> bool:
        if not self.in_bounds(mx, my):
            return True
        return int(self.grid[my][mx]) >= 50

    def cell_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        x = self.origin_x + (float(mx) + 0.5) * self.resolution
        y = self.origin_y + (float(my) + 0.5) * self.resolution
        return x, y


@dataclass
class _ActiveGoal:
    mx: int
    my: int
    start_x: float
    start_y: float
    target_x: float
    target_y: float
    start_yaw_deg: float
    target_yaw_deg: float
    start_ts: float
    duration_s: float


def _strip_quotes(text: str) -> str:
    out = text.strip()
    if (out.startswith("'") and out.endswith("'")) or (out.startswith('"') and out.endswith('"')):
        return out[1:-1]
    return out


def _load_map_yaml(yaml_path: Path) -> tuple[Path, float, tuple[float, float]]:
    data: dict[str, str] = {}
    for raw in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()

    image_field = _strip_quotes(data.get("image", ""))
    if not image_field:
        raise ValueError(f"Map yaml missing 'image': {yaml_path}")
    image_path = (yaml_path.parent / image_field).resolve()

    resolution = float(data.get("resolution", "0.10"))
    origin_field = data.get("origin", "[0.0, 0.0, 0.0]")
    origin = ast.literal_eval(origin_field)
    if not isinstance(origin, (list, tuple)) or len(origin) < 2:
        raise ValueError(f"Map yaml origin is invalid: {origin_field}")
    return image_path, resolution, (float(origin[0]), float(origin[1]))


def _load_pgm_ascii(path: Path) -> tuple[int, int, int, list[int]]:
    tokens: list[str] = []
    with path.open("r", encoding="ascii") as f:
        for raw in f:
            line = raw.split("#", 1)[0]
            tokens.extend(line.split())
    if len(tokens) < 4:
        raise ValueError(f"Invalid PGM: {path}")
    magic = tokens[0]
    if magic != "P2":
        raise ValueError(f"Unsupported PGM format {magic}. Expected P2")
    width = int(tokens[1])
    height = int(tokens[2])
    maxval = int(tokens[3])
    values = [int(v) for v in tokens[4:]]
    if len(values) != width * height:
        raise ValueError(
            f"PGM pixel count mismatch: expected={width * height} got={len(values)} file={path}"
        )
    return width, height, maxval, values


def _load_occupancy_map(yaml_path: Path) -> OccupancyMapSpec:
    image_path, resolution, (origin_x, origin_y) = _load_map_yaml(yaml_path)
    if not image_path.exists():
        raise FileNotFoundError(f"PGM image file not found: {image_path}")

    width, height, maxval, pixels = _load_pgm_ascii(image_path)
    threshold = int(max(1, round(maxval * 0.5)))
    grid = [[0 for _ in range(width)] for _ in range(height)]

    # PGM row 0 starts at top. Internal map grid keeps y-up with row 0 at bottom.
    for my in range(height):
        img_row = (height - 1) - my
        offset = img_row * width
        row = grid[my]
        for mx in range(width):
            pixel = pixels[offset + mx]
            row[mx] = 100 if pixel <= threshold else 0

    return OccupancyMapSpec(
        width=width,
        height=height,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        grid=grid,
    )


class OccupancyCellMoveController:
    """Move robot root by occupancy-cell targets using xform ops only (no physics)."""

    def __init__(
        self,
        stage_obj,
        robot_prim_path: str,
        occupancy: OccupancyMapSpec,
        move_duration_s: float,
        fixed_z: Optional[float],
    ) -> None:
        from pxr import Gf, UsdGeom  # type: ignore

        self._Gf = Gf
        self._UsdGeom = UsdGeom
        self._occupancy = occupancy
        self._move_duration_s = max(0.0, float(move_duration_s))

        prim = stage_obj.GetPrimAtPath(robot_prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Robot prim not found: {robot_prim_path}")

        xform = UsdGeom.Xformable(prim)
        translate_op = get_translate_op(xform)
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
        self._fixed_z = float(pos[2]) if fixed_z is None else float(fixed_z)
        self._translate_op = translate_op
        self._rotate_z_op = rotate_z_op

        self._queue: Deque[Tuple[int, int, Optional[float]]] = deque()
        self._queue_lock = threading.Lock()
        self._active_goal: Optional[_ActiveGoal] = None
        self._last_warn_ts = 0.0

        logging.info(
            "Occupancy controller initialized: target_prim=%s fixed_z=%.3f map=%dx%d res=%.3f",
            robot_prim_path,
            self._fixed_z,
            occupancy.width,
            occupancy.height,
            occupancy.resolution,
        )

    def queue_cell_goal(self, mx: int, my: int, yaw_deg: Optional[float] = None) -> bool:
        if not self._occupancy.in_bounds(mx, my):
            logging.warning("Rejected goal cell=(%d,%d): out of map bounds", mx, my)
            return False
        if self._occupancy.is_occupied(mx, my):
            logging.warning("Rejected goal cell=(%d,%d): occupied cell", mx, my)
            return False

        wx, wy = self._occupancy.cell_to_world(mx, my)
        with self._queue_lock:
            self._queue.append((mx, my, yaw_deg))
        logging.info(
            "Queued occupancy goal: cell=(%d,%d) world=(%.3f, %.3f) yaw=%s",
            mx,
            my,
            wx,
            wy,
            "keep" if yaw_deg is None else f"{yaw_deg:.2f}",
        )
        return True

    def _begin_next_goal(self, now_ts: float) -> None:
        queued: Optional[Tuple[int, int, Optional[float]]] = None
        with self._queue_lock:
            if self._queue:
                queued = self._queue.popleft()
        if queued is None:
            return

        mx, my, yaw_deg = queued
        wx, wy = self._occupancy.cell_to_world(mx, my)
        pos = self._translate_op.Get() or self._Gf.Vec3d(0.0, 0.0, self._fixed_z)
        start_yaw = float(self._rotate_z_op.Get() or 0.0)
        target_yaw = float(yaw_deg) if yaw_deg is not None else start_yaw
        self._active_goal = _ActiveGoal(
            mx=mx,
            my=my,
            start_x=float(pos[0]),
            start_y=float(pos[1]),
            target_x=wx,
            target_y=wy,
            start_yaw_deg=start_yaw,
            target_yaw_deg=target_yaw,
            start_ts=now_ts,
            duration_s=self._move_duration_s,
        )
        logging.info(
            "Start goal: cell=(%d,%d) from=(%.3f,%.3f) to=(%.3f,%.3f) duration=%.2fs",
            mx,
            my,
            float(pos[0]),
            float(pos[1]),
            wx,
            wy,
            self._move_duration_s,
        )

    def step(self, now_ts: float) -> None:
        if self._active_goal is None:
            self._begin_next_goal(now_ts)
            if self._active_goal is None:
                return

        goal = self._active_goal
        if goal is None:
            return

        if goal.duration_s <= 1e-6:
            alpha = 1.0
        else:
            alpha = max(0.0, min(1.0, (now_ts - goal.start_ts) / goal.duration_s))

        next_x = goal.start_x + (goal.target_x - goal.start_x) * alpha
        next_y = goal.start_y + (goal.target_y - goal.start_y) * alpha
        next_yaw = goal.start_yaw_deg + (goal.target_yaw_deg - goal.start_yaw_deg) * alpha
        self._translate_op.Set(self._Gf.Vec3d(next_x, next_y, self._fixed_z))
        self._rotate_z_op.Set(float(next_yaw))

        if alpha >= 1.0:
            logging.info(
                "Goal reached: cell=(%d,%d) world=(%.3f,%.3f)",
                goal.mx,
                goal.my,
                goal.target_x,
                goal.target_y,
            )
            self._active_goal = None

    def print_prompt(self) -> None:
        now = time.time()
        if now - self._last_warn_ts > 3.0:
            self._last_warn_ts = now
            print("Input goal as: <mx> <my> [yaw_deg]  (or 'exit')")


class Ros2GoalSubscriber:
    def __init__(self, topic: str, controller: OccupancyCellMoveController) -> None:
        self._topic = topic
        self._controller = controller
        self._enabled = False
        self._owns_rclpy_init = False
        self._rclpy = None
        self._node = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Point
        except Exception as exc:
            logging.warning("ROS2 occupancy-goal input disabled (imports failed): %s", exc)
            return

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_init = True

        self._node = rclpy.create_node("g1_occupancy_goal_subscriber")
        self._node.create_subscription(Point, self._topic, self._on_point, 20)
        self._enabled = True
        logging.info("ROS2 occupancy-goal topic active: %s (Point.x=mx, y=my, z=yaw_deg)", self._topic)

    def _on_point(self, msg) -> None:
        mx = int(round(float(msg.x)))
        my = int(round(float(msg.y)))
        yaw_deg = float(msg.z)
        self._controller.queue_cell_goal(mx, my, yaw_deg=yaw_deg)

    def spin_once(self) -> None:
        if self._enabled and self._node is not None and self._rclpy is not None:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def stop(self) -> None:
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        self._node = None
        if self._rclpy is not None and self._owns_rclpy_init:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass
        self._enabled = False


class StdinGoalInput:
    def __init__(self, controller: OccupancyCellMoveController) -> None:
        self._controller = controller
        self._shutdown_requested = False
        self._thread = threading.Thread(target=self._run, name="stdin_goal_input", daemon=True)

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                line = input().strip()
            except EOFError:
                return
            if not line:
                continue

            lower = line.lower()
            if lower in {"q", "quit", "exit"}:
                self._shutdown_requested = True
                return

            parts = line.split()
            if len(parts) < 2:
                print("Invalid input. Use: <mx> <my> [yaw_deg]")
                continue
            try:
                mx = int(parts[0])
                my = int(parts[1])
                yaw = float(parts[2]) if len(parts) >= 3 else None
            except ValueError:
                print("Invalid numeric values. Use integers for mx,my and optional float yaw_deg.")
                continue
            self._controller.queue_cell_goal(mx, my, yaw_deg=yaw)


def _parse_goal_cell(raw: str) -> Tuple[int, int]:
    if "," in raw:
        pieces = [p.strip() for p in raw.split(",", 1)]
    else:
        pieces = raw.split()
    if len(pieces) != 2:
        raise ValueError("Goal cell must be 'mx,my' or 'mx my'")
    return int(pieces[0]), int(pieces[1])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-physics occupancy-cell controller in Isaac Sim.",
    )
    parser.add_argument("--stage-usd", required=True, type=str)
    parser.add_argument("--occupancy-map-yaml", required=True, type=str)
    parser.add_argument("--robot-prim-path", default="/World/Robots/G1")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--namespace", default="g1")
    parser.add_argument("--goal-topic", default="")
    parser.add_argument("--disable-ros2-goals", action="store_true")
    parser.add_argument("--enable-stdin-goals", action="store_true")
    parser.add_argument("--start-goal-cell", default="")
    parser.add_argument("--move-duration-s", type=float, default=1.5)
    parser.add_argument("--fixed-z", type=float, default=None)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [occupancy_controller] %(message)s",
    )

    stage_usd = Path(args.stage_usd).resolve()
    map_yaml = Path(args.occupancy_map_yaml).resolve()
    if not stage_usd.exists():
        raise FileNotFoundError(f"Stage USD not found: {stage_usd}")
    if not map_yaml.exists():
        raise FileNotFoundError(f"Occupancy map YAML not found: {map_yaml}")

    occupancy = _load_occupancy_map(map_yaml)
    logging.info(
        "Loaded occupancy map: yaml=%s size=%dx%d res=%.3f origin=(%.3f,%.3f)",
        map_yaml,
        occupancy.width,
        occupancy.height,
        occupancy.resolution,
        occupancy.origin_x,
        occupancy.origin_y,
    )

    from isaacsim import SimulationApp  # type: ignore

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    ros_subscriber: Optional[Ros2GoalSubscriber] = None
    stdin_input: Optional[StdinGoalInput] = None
    try:
        import omni.usd  # type: ignore

        ctx = omni.usd.get_context()
        ok = ctx.open_stage(str(stage_usd))
        if not ok:
            raise RuntimeError(f"Failed to open stage USD: {stage_usd}")

        for _ in range(max(1, int(args.warmup_frames))):
            simulation_app.update()
        stage_obj = ctx.get_stage()
        if stage_obj is None:
            raise RuntimeError("Stage object is not available after open_stage")

        requested_robot_prim = normalize_prim_path(str(args.robot_prim_path), "/World/Robots/G1")
        robot_prim = stage_obj.GetPrimAtPath(requested_robot_prim)
        resolved_robot_prim_path = requested_robot_prim
        if not robot_prim.IsValid():
            fallback = find_robot_prim_path(stage_obj)
            if fallback:
                resolved_robot_prim_path = fallback
                logging.warning(
                    "Requested robot prim not found (%s). Using fallback prim: %s",
                    requested_robot_prim,
                    resolved_robot_prim_path,
                )
            else:
                raise RuntimeError(
                    f"Robot prim is missing and fallback detection failed: {requested_robot_prim}"
                )

        controller = OccupancyCellMoveController(
            stage_obj=stage_obj,
            robot_prim_path=resolved_robot_prim_path,
            occupancy=occupancy,
            move_duration_s=float(args.move_duration_s),
            fixed_z=args.fixed_z,
        )

        if str(args.start_goal_cell).strip():
            mx, my = _parse_goal_cell(str(args.start_goal_cell))
            controller.queue_cell_goal(mx, my, yaw_deg=None)

        topic = str(args.goal_topic).strip()
        if not topic:
            ns = str(args.namespace).strip().strip("/")
            topic = f"/{ns}/cmd/occupancy_goal" if ns else "/cmd/occupancy_goal"

        if not bool(args.disable_ros2_goals):
            ros_subscriber = Ros2GoalSubscriber(topic=topic, controller=controller)
            ros_subscriber.start()

        if bool(args.enable_stdin_goals):
            print("Stdin goal input enabled.")
            controller.print_prompt()
            stdin_input = StdinGoalInput(controller)
            stdin_input.start()

        logging.info(
            "Controller loop started (no physics). Stage=%s robot_prim=%s",
            stage_usd,
            resolved_robot_prim_path,
        )
        while simulation_app.is_running():
            if ros_subscriber is not None and ros_subscriber.enabled:
                ros_subscriber.spin_once()
            controller.step(time.time())
            simulation_app.update()
            if stdin_input is not None and stdin_input.shutdown_requested:
                logging.info("Exit requested via stdin input.")
                break
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        if ros_subscriber is not None:
            ros_subscriber.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()

