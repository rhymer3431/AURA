from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

from .params import CostmapParams, CameraParams


@dataclass
class CostmapResult:
    grid: OccupancyGrid
    origin_xy: Tuple[float, float]
    resolution: float
    data: np.ndarray


class DepthCostmap:
    def __init__(self, costmap_params: CostmapParams, camera_params: CameraParams) -> None:
        self._p = costmap_params
        self._c = camera_params
        self._grid_w = int(math.ceil(costmap_params.size_x / costmap_params.resolution))
        self._grid_h = int(math.ceil(costmap_params.size_y / costmap_params.resolution))
        self._origin_x = -costmap_params.size_x * 0.5
        self._origin_y = -costmap_params.size_y * 0.5

    def _decode_depth(self, depth_img: np.ndarray) -> np.ndarray:
        if self._c.depth_encoding == "16UC1":
            depth_m = depth_img.astype(np.float32) * (self._c.depth_scale / 1000.0)
        else:
            depth_m = depth_img.astype(np.float32) * float(self._c.depth_scale)
        return depth_m

    def _project_to_grid(self, depth: np.ndarray) -> np.ndarray:
        height, width = depth.shape
        stride = max(1, int(self._p.pixel_stride))
        fx = float(self._c.fx)
        fy = float(self._c.fy)
        cx = float(self._c.cx)
        cy = float(self._c.cy)

        grid = np.full((self._grid_h, self._grid_w), -1, dtype=np.int8)

        for v in range(0, height, stride):
            z = depth[v]
            if z.ndim == 0:
                continue
            for u in range(0, width, stride):
                zval = float(z[u])
                if not math.isfinite(zval):
                    continue
                if zval < self._p.min_depth or zval > self._p.max_depth:
                    continue
                x = (u - cx) * zval / fx
                y = (v - cy) * zval / fy
                zc = zval

                height_val = -y  # approximate camera optical frame: y down
                height_above_ground = float(self._p.camera_height) + height_val
                if height_above_ground < self._p.min_height or height_above_ground > self._p.max_height:
                    continue

                # Ground projection: assume obstacle at z forward, x right, y down
                forward = zc
                lateral = -x  # align camera optical +x (right) to base_link +y (left)

                gx = int((forward - self._origin_x) / self._p.resolution)
                gy = int((lateral - self._origin_y) / self._p.resolution)
                if gx < 0 or gy < 0 or gx >= self._grid_w or gy >= self._grid_h:
                    continue
                grid[gy, gx] = 100

        return grid

    def _inflate(self, grid: np.ndarray) -> np.ndarray:
        radius = max(0.0, float(self._p.inflation_radius))
        if radius <= 0.0:
            return grid
        cells = int(math.ceil(radius / self._p.resolution))
        if cells <= 0:
            return grid
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cells * 2 + 1, cells * 2 + 1))
        obstacle = (grid == 100).astype(np.uint8)
        inflated = cv2.dilate(obstacle, kernel)
        out = grid.copy()
        out[inflated > 0] = 100
        return out

    def build_costmap(self, depth_img: np.ndarray, header: Optional[Header] = None) -> CostmapResult:
        depth_m = self._decode_depth(depth_img)
        grid = self._project_to_grid(depth_m)
        grid = self._inflate(grid)

        if self._p.unknown_as_obstacle:
            grid[grid < 0] = 100
        else:
            grid[grid < 0] = 0

        msg = OccupancyGrid()
        if header is not None:
            msg.header = header
        msg.info.resolution = float(self._p.resolution)
        msg.info.width = int(self._grid_w)
        msg.info.height = int(self._grid_h)
        msg.info.origin.position.x = float(self._origin_x)
        msg.info.origin.position.y = float(self._origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.astype(np.int8).flatten().tolist()

        return CostmapResult(msg, (self._origin_x, self._origin_y), self._p.resolution, grid)

    def check_collision_ahead(self, depth_img: np.ndarray) -> bool:
        depth_m = self._decode_depth(depth_img)
        height, width = depth_m.shape
        fov = math.radians(self._p.safety_fov_deg)
        cx = float(self._c.cx)
        fx = float(self._c.fx)

        max_angle = fov * 0.5
        for v in range(0, height, max(1, int(self._p.pixel_stride))):
            for u in range(0, width, max(1, int(self._p.pixel_stride))):
                z = float(depth_m[v, u])
                if not math.isfinite(z):
                    continue
                if z < self._p.min_depth or z > self._p.max_depth:
                    continue
                angle = math.atan2((u - cx), fx)
                if abs(angle) > max_angle:
                    continue
                if z <= self._p.safety_stop_distance:
                    return True
        return False
