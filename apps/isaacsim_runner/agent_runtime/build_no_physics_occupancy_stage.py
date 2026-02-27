from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.isaacsim_runner.config.base import DEFAULT_G1_START_Z


DEFAULT_G1_USD = (
    ROOT_DIR
    / "apps"
    / "isaac_ros2_bridge_bundle"
    / "robot_model"
    / "model_data"
    / "g1"
    / "g1_29dof_with_hand"
    / "g1_29dof_with_hand.usd"
)

DEFAULT_STAGE_OUT = ROOT_DIR / "tmp" / "agent_runtime" / "g1_no_physics_stage.usda"
DEFAULT_MAP_DIR = ROOT_DIR / "tmp" / "agent_runtime" / "maps"

DEFAULT_OBSTACLES: List[Dict[str, Any]] = [
    {"name": "wall_left", "center": [2.8, 0.0], "size": [0.4, 6.0, 1.2]},
    {"name": "wall_right", "center": [-2.8, 0.0], "size": [0.4, 6.0, 1.2]},
    {"name": "table_block", "center": [0.0, 2.3], "size": [1.8, 0.8, 0.9]},
]


@dataclass
class ObstacleSpec:
    name: str
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    size_z: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-physics Isaac stage with G1 USD and export occupancy map artifacts.",
    )
    parser.add_argument("--g1-usd", type=str, default=str(DEFAULT_G1_USD))
    parser.add_argument("--stage-out", type=str, default=str(DEFAULT_STAGE_OUT))
    parser.add_argument("--map-dir", type=str, default=str(DEFAULT_MAP_DIR))
    parser.add_argument("--map-name", type=str, default="g1_no_physics")
    parser.add_argument("--resolution", type=float, default=0.10, help="Meters per cell")
    parser.add_argument("--map-width-m", type=float, default=20.0)
    parser.add_argument("--map-height-m", type=float, default=20.0)
    parser.add_argument("--origin-x", type=float, default=-10.0)
    parser.add_argument("--origin-y", type=float, default=-10.0)
    parser.add_argument("--inflate-cells", type=int, default=1)
    parser.add_argument("--robot-prim-path", type=str, default="/World/Robots/G1")
    parser.add_argument("--robot-start-x", type=float, default=0.0)
    parser.add_argument("--robot-start-y", type=float, default=0.0)
    parser.add_argument("--robot-start-z", type=float, default=DEFAULT_G1_START_Z)
    parser.add_argument("--robot-start-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--obstacles-json",
        type=str,
        default="",
        help="Optional obstacle definition JSON. Supports list[...] or {'obstacles': [...]}",
    )
    parser.add_argument("--headless", action="store_true", help="Run stage builder in headless mode")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _load_obstacles(path: Path | None) -> List[ObstacleSpec]:
    payload: Sequence[Dict[str, Any]]
    if path is None:
        payload = DEFAULT_OBSTACLES
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("obstacles", [])
        if not isinstance(data, list):
            raise ValueError("Obstacle JSON must be a list or {'obstacles': list}")
        payload = data

    out: List[ObstacleSpec] = []
    for idx, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"Obstacle at index {idx} is not an object")
        name = str(raw.get("name", f"obs_{idx:02d}")).strip() or f"obs_{idx:02d}"
        center = raw.get("center", [0.0, 0.0])
        size = raw.get("size", [1.0, 1.0, 1.0])
        if not isinstance(center, list) or len(center) < 2:
            raise ValueError(f"Obstacle '{name}' center must contain [x, y]")
        if not isinstance(size, list) or len(size) < 2:
            raise ValueError(f"Obstacle '{name}' size must contain [sx, sy, (sz)]")

        sx = float(size[0])
        sy = float(size[1])
        sz = float(size[2]) if len(size) >= 3 else 1.0
        if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
            raise ValueError(f"Obstacle '{name}' has non-positive size")
        out.append(
            ObstacleSpec(
                name=name,
                center_x=float(center[0]),
                center_y=float(center[1]),
                size_x=sx,
                size_y=sy,
                size_z=sz,
            )
        )
    return out


def _sanitize_token(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            keep.append(ch)
        else:
            keep.append("_")
    token = "".join(keep).strip("_")
    return token or "obstacle"


def _set_translate_and_yaw(prim, xyz: tuple[float, float, float], yaw_deg: float) -> None:
    from pxr import Gf, UsdGeom  # type: ignore

    xform = UsdGeom.Xformable(prim)
    translate_op = None
    rotate_op = None
    for op in xform.GetOrderedXformOps():
        if translate_op is None and op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        if rotate_op is None and op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
            rotate_op = op
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    if rotate_op is None:
        rotate_op = xform.AddRotateZOp()
    translate_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    rotate_op.Set(float(yaw_deg))


def _build_stage(args: argparse.Namespace, g1_usd: Path, obstacles: Sequence[ObstacleSpec], stage_out: Path) -> None:
    from isaacsim import SimulationApp  # type: ignore

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        import omni.usd  # type: ignore
        from pxr import Gf, UsdGeom, UsdLux  # type: ignore

        ctx = omni.usd.get_context()
        ctx.new_stage()
        for _ in range(20):
            simulation_app.update()
        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError("Failed to create new USD stage")

        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        UsdGeom.Xform.Define(stage, "/World/Environment")
        UsdGeom.Xform.Define(stage, "/World/Environment/Obstacles")
        UsdGeom.Xform.Define(stage, "/World/Robots")
        stage.SetDefaultPrim(world)

        key_light = UsdLux.DistantLight.Define(stage, "/World/Environment/KeyLight")
        key_light.CreateIntensityAttr(900.0)
        key_light.CreateAngleAttr(0.53)

        ground = UsdGeom.Cube.Define(stage, "/World/Environment/Ground")
        ground.CreateSizeAttr(1.0)
        _set_translate_and_yaw(
            ground.GetPrim(),
            (0.0, 0.0, -0.02),
            0.0,
        )
        ground_xf = UsdGeom.Xformable(ground.GetPrim())
        scale_op = None
        for op in ground_xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op is None:
            scale_op = ground_xf.AddScaleOp()
        scale_op.Set(Gf.Vec3f(float(args.map_width_m), float(args.map_height_m), 0.02))
        UsdGeom.Gprim(ground.GetPrim()).CreateDisplayColorAttr([(0.20, 0.22, 0.24)])

        robot_prim = UsdGeom.Xform.Define(stage, str(args.robot_prim_path)).GetPrim()
        robot_prim.GetReferences().AddReference(str(g1_usd))
        _set_translate_and_yaw(
            robot_prim,
            (float(args.robot_start_x), float(args.robot_start_y), float(args.robot_start_z)),
            float(args.robot_start_yaw_deg),
        )

        for idx, obs in enumerate(obstacles):
            prim_path = f"/World/Environment/Obstacles/{_sanitize_token(obs.name)}_{idx:02d}"
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)
            _set_translate_and_yaw(
                cube.GetPrim(),
                (obs.center_x, obs.center_y, obs.size_z * 0.5),
                0.0,
            )
            xf = UsdGeom.Xformable(cube.GetPrim())
            scale_op = None
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                    break
            if scale_op is None:
                scale_op = xf.AddScaleOp()
            scale_op.Set(Gf.Vec3f(obs.size_x, obs.size_y, obs.size_z))
            UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([(0.90, 0.48, 0.18)])

        stage.GetRootLayer().Export(str(stage_out))
        logging.info("Saved no-physics stage: %s", stage_out)
    finally:
        simulation_app.close()


def _build_grid(
    obstacles: Sequence[ObstacleSpec],
    width_cells: int,
    height_cells: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    inflate_cells: int,
) -> List[List[int]]:
    grid = [[0 for _ in range(width_cells)] for _ in range(height_cells)]

    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    for obs in obstacles:
        min_x = obs.center_x - (obs.size_x * 0.5)
        max_x = obs.center_x + (obs.size_x * 0.5)
        min_y = obs.center_y - (obs.size_y * 0.5)
        max_y = obs.center_y + (obs.size_y * 0.5)

        ix0 = int(math.floor((min_x - origin_x) / resolution)) - inflate_cells
        ix1 = int(math.ceil((max_x - origin_x) / resolution)) - 1 + inflate_cells
        iy0 = int(math.floor((min_y - origin_y) / resolution)) - inflate_cells
        iy1 = int(math.ceil((max_y - origin_y) / resolution)) - 1 + inflate_cells

        ix0 = clamp(ix0, 0, width_cells - 1)
        ix1 = clamp(ix1, 0, width_cells - 1)
        iy0 = clamp(iy0, 0, height_cells - 1)
        iy1 = clamp(iy1, 0, height_cells - 1)
        if ix0 > ix1 or iy0 > iy1:
            continue

        for iy in range(iy0, iy1 + 1):
            row = grid[iy]
            for ix in range(ix0, ix1 + 1):
                row[ix] = 100

    # Reserve boundaries as occupied for safer map-coordinate goals.
    for ix in range(width_cells):
        grid[0][ix] = 100
        grid[height_cells - 1][ix] = 100
    for iy in range(height_cells):
        grid[iy][0] = 100
        grid[iy][width_cells - 1] = 100
    return grid


def _write_pgm(path: Path, grid: Sequence[Sequence[int]]) -> None:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    with path.open("w", encoding="ascii") as f:
        f.write("P2\n")
        f.write(f"{width} {height}\n")
        f.write("255\n")
        for my in range(height - 1, -1, -1):
            vals: List[str] = []
            for mx in range(width):
                occ = int(grid[my][mx])
                vals.append("0" if occ >= 50 else "254")
            f.write(" ".join(vals))
            f.write("\n")


def _write_yaml(path: Path, map_name: str, resolution: float, origin_x: float, origin_y: float) -> None:
    text = (
        f"image: {map_name}.pgm\n"
        f"resolution: {resolution:.6f}\n"
        f"origin: [{origin_x:.6f}, {origin_y:.6f}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )
    path.write_text(text, encoding="utf-8")


def _write_metadata_json(
    path: Path,
    args: argparse.Namespace,
    stage_out: Path,
    map_yaml: Path,
    width_cells: int,
    height_cells: int,
    obstacles: Sequence[ObstacleSpec],
) -> None:
    payload = {
        "stage_usd": str(stage_out),
        "map_yaml": str(map_yaml),
        "resolution": float(args.resolution),
        "origin": [float(args.origin_x), float(args.origin_y), 0.0],
        "size_cells": [int(width_cells), int(height_cells)],
        "robot_prim_path": str(args.robot_prim_path),
        "robot_start": [
            float(args.robot_start_x),
            float(args.robot_start_y),
            float(args.robot_start_z),
            float(args.robot_start_yaw_deg),
        ],
        "obstacles": [
            {
                "name": obs.name,
                "center": [obs.center_x, obs.center_y],
                "size": [obs.size_x, obs.size_y, obs.size_z],
            }
            for obs in obstacles
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [build_no_physics_stage] %(message)s",
    )

    g1_usd = Path(args.g1_usd).resolve()
    if not g1_usd.exists():
        raise FileNotFoundError(f"G1 USD not found: {g1_usd}")

    stage_out = Path(args.stage_out).resolve()
    stage_out.parent.mkdir(parents=True, exist_ok=True)

    map_dir = Path(args.map_dir).resolve()
    map_dir.mkdir(parents=True, exist_ok=True)

    obstacles_path: Path | None = None
    if str(args.obstacles_json).strip():
        obstacles_path = Path(args.obstacles_json).resolve()
        if not obstacles_path.exists():
            raise FileNotFoundError(f"Obstacles JSON not found: {obstacles_path}")
    obstacles = _load_obstacles(obstacles_path)

    width_cells = max(1, int(math.ceil(float(args.map_width_m) / float(args.resolution))))
    height_cells = max(1, int(math.ceil(float(args.map_height_m) / float(args.resolution))))
    if width_cells < 2 or height_cells < 2:
        raise ValueError("Map dimensions are too small after resolution conversion")

    _build_stage(args, g1_usd=g1_usd, obstacles=obstacles, stage_out=stage_out)

    grid = _build_grid(
        obstacles=obstacles,
        width_cells=width_cells,
        height_cells=height_cells,
        resolution=float(args.resolution),
        origin_x=float(args.origin_x),
        origin_y=float(args.origin_y),
        inflate_cells=max(0, int(args.inflate_cells)),
    )

    map_name = str(args.map_name).strip() or "g1_no_physics"
    map_pgm = map_dir / f"{map_name}.pgm"
    map_yaml = map_dir / f"{map_name}.yaml"
    meta_json = map_dir / f"{map_name}.meta.json"

    _write_pgm(map_pgm, grid)
    _write_yaml(map_yaml, map_name, float(args.resolution), float(args.origin_x), float(args.origin_y))
    _write_metadata_json(meta_json, args, stage_out, map_yaml, width_cells, height_cells, obstacles)

    occupied = sum(1 for row in grid for v in row if v >= 50)
    total = width_cells * height_cells
    logging.info("Occupancy map exported: yaml=%s pgm=%s", map_yaml, map_pgm)
    logging.info(
        "Map summary: cells=%dx%d occupied=%d free=%d",
        width_cells,
        height_cells,
        occupied,
        total - occupied,
    )
    print(f"STAGE_USD={stage_out}")
    print(f"MAP_YAML={map_yaml}")
    print(f"MAP_META={meta_json}")


if __name__ == "__main__":
    main()

