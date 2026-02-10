from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path

from .params import PlannerParams


class AStarPlanner:
    def __init__(self, params: PlannerParams) -> None:
        self._p = params

    def _grid_from_msg(self, grid: OccupancyGrid) -> np.ndarray:
        data = np.array(grid.data, dtype=np.int16)
        return data.reshape((grid.info.height, grid.info.width))

    def _is_free(self, grid: np.ndarray, ix: int, iy: int) -> bool:
        val = grid[iy, ix]
        if val < 0:
            return not self._p.unknown_as_obstacle
        return val < self._p.occupied_threshold

    def _neighbors(self) -> List[Tuple[int, int, float]]:
        if self._p.allow_diagonal:
            return [
                (-1, 0, 1.0),
                (1, 0, 1.0),
                (0, -1, 1.0),
                (0, 1, 1.0),
                (-1, -1, math.sqrt(2.0)),
                (-1, 1, math.sqrt(2.0)),
                (1, -1, math.sqrt(2.0)),
                (1, 1, math.sqrt(2.0)),
            ]
        return [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
        ]

    def plan(
        self, grid_msg: OccupancyGrid, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]
    ) -> Path:
        path_msg = Path()
        if grid_msg is None:
            return path_msg
        path_msg.header = grid_msg.header

        grid = self._grid_from_msg(grid_msg)
        res = float(grid_msg.info.resolution)
        origin_x = float(grid_msg.info.origin.position.x)
        origin_y = float(grid_msg.info.origin.position.y)
        width = int(grid_msg.info.width)
        height = int(grid_msg.info.height)

        def to_cell(x: float, y: float) -> Optional[Tuple[int, int]]:
            ix = int((x - origin_x) / res)
            iy = int((y - origin_y) / res)
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                return None
            return ix, iy

        start_cell = to_cell(start_xy[0], start_xy[1])
        goal_cell = to_cell(goal_xy[0], goal_xy[1])
        if start_cell is None or goal_cell is None:
            return path_msg

        if not self._is_free(grid, start_cell[0], start_cell[1]):
            return path_msg
        if not self._is_free(grid, goal_cell[0], goal_cell[1]):
            return path_msg

        neighbors = self._neighbors()
        open_heap: List[Tuple[float, float, int, int]] = []
        g_score: Dict[Tuple[int, int], float] = {start_cell: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        heapq.heappush(open_heap, (heuristic(start_cell, goal_cell), 0.0, start_cell[0], start_cell[1]))

        iterations = 0
        found = False
        while open_heap and iterations < int(self._p.max_iterations):
            iterations += 1
            _, g, cx, cy = heapq.heappop(open_heap)
            current = (cx, cy)
            if current == goal_cell:
                found = True
                break

            for dx, dy, cost in neighbors:
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if not self._is_free(grid, nx, ny):
                    continue
                ng = g + cost
                neighbor = (nx, ny)
                if neighbor not in g_score or ng < g_score[neighbor]:
                    g_score[neighbor] = ng
                    came_from[neighbor] = current
                    f = ng + heuristic(neighbor, goal_cell)
                    heapq.heappush(open_heap, (f, ng, nx, ny))

        if not found:
            return path_msg

        cells = [goal_cell]
        while cells[-1] != start_cell:
            parent = came_from.get(cells[-1])
            if parent is None:
                break
            cells.append(parent)
        cells.reverse()

        poses: List[PoseStamped] = []
        for i, (ix, iy) in enumerate(cells):
            x = origin_x + (ix + 0.5) * res
            y = origin_y + (iy + 0.5) * res
            pose = PoseStamped()
            pose.header = grid_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            yaw = 0.0
            if i + 1 < len(cells):
                nx, ny = cells[i + 1]
                nx_w = origin_x + (nx + 0.5) * res
                ny_w = origin_y + (ny + 0.5) * res
                yaw = math.atan2(ny_w - y, nx_w - x)
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            poses.append(pose)

        path_msg.poses = poses
        return path_msg
