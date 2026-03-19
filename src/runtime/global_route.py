from __future__ import annotations

import math
import re
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
from PIL import Image

from common.cv2_compat import cv2

GridPoint = tuple[int, int]
WorldPoint = tuple[float, float]

_TOP_PATTERN = re.compile(
    r"Top Left:\s*\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)\s*Top Right:\s*\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)",
    re.MULTILINE,
)
_BOTTOM_PATTERN = re.compile(
    r"Bottom Left:\s*\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)\s*Bottom Right:\s*\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)",
    re.MULTILINE,
)
_SIZE_PATTERN = re.compile(r"Image size in pixels:\s*(\d+)\s*,\s*(\d+)", re.MULTILINE)


@dataclass(frozen=True)
class MapMeta:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: int
    height: int
    res: float

    @classmethod
    def from_bounds(
        cls,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        width: int,
        height: int,
    ) -> MapMeta:
        width_px = int(width)
        height_px = int(height)
        if width_px < 2 or height_px < 2:
            raise ValueError(f"map image must be at least 2x2, got width={width_px} height={height_px}")
        res_x = (float(x_max) - float(x_min)) / float(width_px - 1)
        res_y = (float(y_max) - float(y_min)) / float(height_px - 1)
        if abs(res_x - res_y) > 1.0e-3:
            raise ValueError(f"resolution mismatch: res_x={res_x:.6f} res_y={res_y:.6f}")
        return cls(
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
            width=width_px,
            height=height_px,
            res=float((res_x + res_y) * 0.5),
        )

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> MapMeta:
        text = Path(config_path).read_text(encoding="utf-8")
        top_match = _TOP_PATTERN.search(text)
        bottom_match = _BOTTOM_PATTERN.search(text)
        size_match = _SIZE_PATTERN.search(text)
        if top_match is None or bottom_match is None or size_match is None:
            raise ValueError(f"invalid global map config format: {config_path}")
        top_left_x, top_left_y, top_right_x, top_right_y = (float(value) for value in top_match.groups())
        bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y = (float(value) for value in bottom_match.groups())
        width, height = (int(value) for value in size_match.groups())
        if abs(top_left_y - top_right_y) > 1.0e-6:
            raise ValueError(f"top corners must share y coordinate: {config_path}")
        if abs(bottom_left_y - bottom_right_y) > 1.0e-6:
            raise ValueError(f"bottom corners must share y coordinate: {config_path}")
        if abs(top_left_x - bottom_left_x) > 1.0e-6:
            raise ValueError(f"left corners must share x coordinate: {config_path}")
        if abs(top_right_x - bottom_right_x) > 1.0e-6:
            raise ValueError(f"right corners must share x coordinate: {config_path}")
        return cls.from_bounds(
            x_min=top_left_x,
            x_max=top_right_x,
            y_min=bottom_left_y,
            y_max=top_left_y,
            width=width,
            height=height,
        )

    def world_to_grid(self, x_world: float, y_world: float, *, clamp: bool = False) -> GridPoint:
        col = int(round((float(x_world) - self.x_min) / self.res))
        row = int(round((self.y_max - float(y_world)) / self.res))
        if clamp:
            col = max(0, min(self.width - 1, col))
            row = max(0, min(self.height - 1, row))
            return row, col
        if not (0 <= col < self.width and 0 <= row < self.height):
            raise ValueError(
                f"world point ({float(x_world):.4f}, {float(y_world):.4f}) is outside map bounds "
                f"x=[{self.x_min:.4f}, {self.x_max:.4f}] y=[{self.y_min:.4f}, {self.y_max:.4f}]"
            )
        return row, col

    def grid_to_world(self, row: int, col: int) -> WorldPoint:
        row_i = int(row)
        col_i = int(col)
        if not (0 <= row_i < self.height and 0 <= col_i < self.width):
            raise ValueError(f"grid point ({row_i}, {col_i}) is outside map size {self.height}x{self.width}")
        x_world = self.x_min + float(col_i) * self.res
        y_world = self.y_max - float(row_i) * self.res
        return float(x_world), float(y_world)


@dataclass(frozen=True)
class GlobalRoutePlanner:
    meta: MapMeta
    occupancy_grid: np.ndarray
    image_path: Path
    config_path: Path
    inflation_radius_px: int

    @classmethod
    def from_files(
        cls,
        *,
        image_path: str | Path,
        config_path: str | Path,
        inflation_radius_m: float,
    ) -> GlobalRoutePlanner:
        image = Path(image_path)
        config = Path(config_path)
        meta = MapMeta.from_config_file(config)
        occupancy = load_occupancy_grid(image, meta=meta)
        inflate_px = int(math.ceil(float(max(inflation_radius_m, 0.0)) / meta.res))
        inflated = inflate_occupancy_grid(occupancy, radius_px=inflate_px)
        return cls(
            meta=meta,
            occupancy_grid=inflated,
            image_path=image,
            config_path=config,
            inflation_radius_px=inflate_px,
        )

    def plan(
        self,
        *,
        start_xy: WorldPoint,
        goal_xy: WorldPoint,
        waypoint_spacing_m: float,
        initial_skip_distance_m: float = 0.15,
    ) -> list[WorldPoint]:
        spacing_m = float(waypoint_spacing_m)
        if spacing_m <= 0.0:
            raise ValueError(f"waypoint_spacing_m must be positive, got {waypoint_spacing_m}")
        start_rc = self.meta.world_to_grid(start_xy[0], start_xy[1], clamp=False)
        goal_rc = self.meta.world_to_grid(goal_xy[0], goal_xy[1], clamp=False)
        if self.occupancy_grid[start_rc] != 0:
            raise ValueError(f"global route start cell is blocked: {start_rc}")
        if self.occupancy_grid[goal_rc] != 0:
            raise ValueError(f"global route goal cell is blocked: {goal_rc}")
        grid_path = theta_star(self.occupancy_grid, start_rc, goal_rc)
        world_path = [self.meta.grid_to_world(row, col) for row, col in grid_path]
        waypoints = resample_polyline(world_path, spacing_m=spacing_m)
        if len(waypoints) == 0:
            waypoints = [(float(goal_xy[0]), float(goal_xy[1]))]
        else:
            waypoints[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        if (
            len(waypoints) >= 2
            and math.hypot(waypoints[0][0] - float(start_xy[0]), waypoints[0][1] - float(start_xy[1]))
            < float(initial_skip_distance_m)
        ):
            waypoints = waypoints[1:]
        if len(waypoints) == 0:
            waypoints = [(float(goal_xy[0]), float(goal_xy[1]))]
        return waypoints


def load_occupancy_grid(image_path: str | Path, *, meta: MapMeta) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    image_array = np.asarray(image, dtype=np.uint8)
    expected_shape = (int(meta.height), int(meta.width))
    if image_array.shape != expected_shape:
        raise ValueError(
            f"occupancy image shape mismatch: expected={expected_shape} actual={tuple(image_array.shape)} path={image_path}"
        )
    return np.where(image_array == 255, 0, 1).astype(np.uint8)


def inflate_occupancy_grid(occupancy_grid: np.ndarray, *, radius_px: int) -> np.ndarray:
    occupancy = np.asarray(occupancy_grid, dtype=np.uint8)
    radius = int(radius_px)
    if occupancy.ndim != 2:
        raise ValueError(f"occupancy_grid must be rank-2, got shape={occupancy.shape}")
    if radius <= 0 or not np.any(occupancy > 0):
        return occupancy.copy()
    kernel = _disk_kernel(radius)
    if hasattr(cv2, "dilate"):
        return cv2.dilate(occupancy, kernel, iterations=1).astype(np.uint8)
    return _manual_dilate(occupancy, kernel)


def theta_star(occupancy_grid: np.ndarray, start: GridPoint, goal: GridPoint) -> list[GridPoint]:
    occupancy = np.asarray(occupancy_grid, dtype=np.uint8)
    if occupancy.ndim != 2:
        raise ValueError(f"occupancy_grid must be rank-2, got shape={occupancy.shape}")
    height, width = occupancy.shape

    def valid(row: int, col: int) -> bool:
        return 0 <= row < height and 0 <= col < width and occupancy[row, col] == 0

    if not valid(*start):
        raise ValueError(f"start is occupied or out of bounds: {start}")
    if not valid(*goal):
        raise ValueError(f"goal is occupied or out of bounds: {goal}")

    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    frontier: list[tuple[float, GridPoint]] = []
    heappush(frontier, (0.0, start))
    parent: dict[GridPoint, GridPoint] = {start: start}
    g_score: dict[GridPoint, float] = {start: 0.0}
    closed: set[GridPoint] = set()

    while frontier:
        _, current = heappop(frontier)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct_path(parent, current)
        closed.add(current)
        current_row, current_col = current
        for d_row, d_col, move_cost in neighbors:
            next_row = current_row + d_row
            next_col = current_col + d_col
            if not valid(next_row, next_col):
                continue
            if d_row != 0 and d_col != 0:
                if not valid(current_row + d_row, current_col) or not valid(current_row, current_col + d_col):
                    continue
            neighbor = (next_row, next_col)
            anchor = parent[current]
            if _line_of_sight(occupancy, anchor, neighbor):
                tentative = g_score[anchor] + math.hypot(neighbor[0] - anchor[0], neighbor[1] - anchor[1])
                candidate_parent = anchor
            else:
                tentative = g_score[current] + move_cost
                candidate_parent = current
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            parent[neighbor] = candidate_parent
            g_score[neighbor] = tentative
            f_score = tentative + math.hypot(goal[0] - next_row, goal[1] - next_col)
            heappush(frontier, (f_score, neighbor))
    raise RuntimeError("No path found")


def a_star(occupancy_grid: np.ndarray, start: GridPoint, goal: GridPoint) -> list[GridPoint]:
    return theta_star(occupancy_grid, start, goal)


def resample_polyline(points: list[WorldPoint], *, spacing_m: float) -> list[WorldPoint]:
    if len(points) <= 1:
        return [(float(x), float(y)) for x, y in points]
    spacing = float(spacing_m)
    if spacing <= 0.0:
        raise ValueError(f"spacing_m must be positive, got {spacing_m}")
    samples = [np.asarray(points[0], dtype=np.float64)]
    accumulated = 0.0
    previous = np.asarray(points[0], dtype=np.float64)
    for current_raw in points[1:]:
        current = np.asarray(current_raw, dtype=np.float64)
        segment = current - previous
        segment_len = float(np.linalg.norm(segment))
        if segment_len < 1.0e-8:
            previous = current
            continue
        direction = segment / segment_len
        remaining = segment_len
        while accumulated + remaining >= spacing:
            step = spacing - accumulated
            new_point = previous + direction * step
            samples.append(new_point)
            previous = new_point
            segment = current - previous
            remaining = float(np.linalg.norm(segment))
            if remaining < 1.0e-8:
                accumulated = 0.0
                break
            direction = segment / remaining
            accumulated = 0.0
        accumulated += remaining
        previous = current
    last_point = np.asarray(points[-1], dtype=np.float64)
    if float(np.linalg.norm(samples[-1] - last_point)) > 1.0e-6:
        samples.append(last_point)
    return [(float(point[0]), float(point[1])) for point in samples]


def _disk_kernel(radius_px: int) -> np.ndarray:
    radius = int(radius_px)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((xx * xx) + (yy * yy)) <= (radius * radius)
    return kernel.astype(np.uint8)


def _manual_dilate(occupancy: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    rows, cols = occupancy.shape
    radius_row = kernel.shape[0] // 2
    radius_col = kernel.shape[1] // 2
    result = np.zeros_like(occupancy, dtype=np.uint8)
    offsets = np.argwhere(kernel > 0)
    for kernel_row, kernel_col in offsets:
        d_row = int(kernel_row) - radius_row
        d_col = int(kernel_col) - radius_col
        src_row_start = max(0, -d_row)
        src_row_end = min(rows, rows - d_row)
        src_col_start = max(0, -d_col)
        src_col_end = min(cols, cols - d_col)
        dst_row_start = max(0, d_row)
        dst_row_end = min(rows, rows + d_row)
        dst_col_start = max(0, d_col)
        dst_col_end = min(cols, cols + d_col)
        result[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = np.maximum(
            result[dst_row_start:dst_row_end, dst_col_start:dst_col_end],
            occupancy[src_row_start:src_row_end, src_col_start:src_col_end],
        )
    return result


def _reconstruct_path(came_from: dict[GridPoint, GridPoint], current: GridPoint) -> list[GridPoint]:
    path = [current]
    node = current
    while node in came_from:
        node = came_from[node]
        if node == path[-1]:
            break
        path.append(node)
    path.reverse()
    return path


def _line_of_sight(occupancy_grid: np.ndarray, start: GridPoint, goal: GridPoint) -> bool:
    occupancy = np.asarray(occupancy_grid, dtype=np.uint8)
    if occupancy.ndim != 2:
        return False
    steps = max(abs(goal[0] - start[0]), abs(goal[1] - start[1]))
    if steps == 0:
        return bool(occupancy[start] == 0)

    prev_row = int(start[0])
    prev_col = int(start[1])
    for index in range(1, steps + 1):
        ratio = float(index) / float(steps)
        row = int(round(start[0] + (goal[0] - start[0]) * ratio))
        col = int(round(start[1] + (goal[1] - start[1]) * ratio))
        if not (0 <= row < occupancy.shape[0] and 0 <= col < occupancy.shape[1]) or occupancy[row, col] != 0:
            return False
        if row != prev_row and col != prev_col:
            if occupancy[row, prev_col] != 0 or occupancy[prev_row, col] != 0:
                return False
        prev_row = row
        prev_col = col
    return True


__all__ = [
    "GlobalRoutePlanner",
    "GridPoint",
    "MapMeta",
    "WorldPoint",
    "a_star",
    "theta_star",
    "inflate_occupancy_grid",
    "load_occupancy_grid",
    "resample_polyline",
]
