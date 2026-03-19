from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime.global_route import GlobalRoutePlanner, MapMeta, inflate_occupancy_grid, load_occupancy_grid, theta_star


def _write_map_assets(tmp_path: Path, image_array: np.ndarray, *, x_max: float | None = None, y_max: float | None = None) -> tuple[Path, Path]:
    image_path = tmp_path / "occupancy_map.png"
    config_path = tmp_path / "config.txt"
    Image.fromarray(np.asarray(image_array, dtype=np.uint8), mode="L").save(image_path)
    height, width = image_array.shape
    max_x = float(width - 1 if x_max is None else x_max)
    max_y = float(height - 1 if y_max is None else y_max)
    config_path.write_text(
        "\n".join(
            [
                f"Top Left: (0.0, {max_y})\t\t Top Right: ({max_x}, {max_y})",
                f" Bottom Left: (0.0, 0.0)\t\t Bottom Right: ({max_x}, 0.0)",
                "Coordinates of top left of image (pixel 0,0) as origin, + X down, + Y right:",
                f"(0.0, {max_x})",
                f"Image size in pixels: {width}, {height}",
            ]
        ),
        encoding="utf-8",
    )
    return image_path, config_path


def test_map_meta_from_kujiale_config_matches_corner_transform() -> None:
    config_path = ROOT / "datasets" / "InteriorAgent" / "kujiale_0003" / "config.txt"
    meta = MapMeta.from_config_file(config_path)

    assert meta.width == 328
    assert meta.height == 281
    assert meta.res == pytest.approx(0.05, abs=1.0e-6)
    assert meta.world_to_grid(meta.x_min, meta.y_max) == (0, 0)
    assert meta.world_to_grid(meta.x_max, meta.y_min) == (280, 327)
    assert meta.world_to_grid(meta.x_min + meta.res, meta.y_max) == (0, 1)
    assert meta.world_to_grid(meta.x_min, meta.y_max - meta.res) == (1, 0)
    assert meta.grid_to_world(0, 0) == pytest.approx((meta.x_min, meta.y_max))


def test_load_occupancy_grid_marks_unknown_as_blocked_and_applies_inflation(tmp_path: Path) -> None:
    image = np.asarray(
        [
            [255, 255, 255],
            [255, 127, 255],
            [255, 0, 255],
        ],
        dtype=np.uint8,
    )
    image_path, config_path = _write_map_assets(tmp_path, image)
    meta = MapMeta.from_config_file(config_path)

    occupancy = load_occupancy_grid(image_path, meta=meta)
    inflated = inflate_occupancy_grid(occupancy, radius_px=1)

    assert occupancy.tolist() == [[0, 0, 0], [0, 1, 0], [0, 1, 0]]
    assert inflated[1, 0] == 1
    assert inflated[1, 1] == 1
    assert inflated[2, 2] == 1


def test_theta_star_rejects_diagonal_corner_cutting() -> None:
    occupancy = np.zeros((3, 3), dtype=np.uint8)
    occupancy[0, 1] = 1
    occupancy[1, 0] = 1

    with pytest.raises(RuntimeError, match="No path found"):
        theta_star(occupancy, (0, 0), (1, 1))


def test_theta_star_rejects_blocked_start() -> None:
    occupancy = np.zeros((3, 3), dtype=np.uint8)
    occupancy[0, 0] = 1

    with pytest.raises(ValueError, match="start is occupied"):
        theta_star(occupancy, (0, 0), (2, 2))


def test_global_route_planner_resamples_and_keeps_exact_final_goal(tmp_path: Path) -> None:
    image = np.full((5, 5), 255, dtype=np.uint8)
    image_path, config_path = _write_map_assets(tmp_path, image)
    planner = GlobalRoutePlanner.from_files(image_path=image_path, config_path=config_path, inflation_radius_m=0.0)

    waypoints = planner.plan(start_xy=(0.0, 0.0), goal_xy=(4.0, 0.0), waypoint_spacing_m=1.0)

    assert waypoints[-1] == pytest.approx((4.0, 0.0))
    assert waypoints[0] == pytest.approx((1.0, 0.0))
    spacings = [math.hypot(waypoints[i + 1][0] - waypoints[i][0], waypoints[i + 1][1] - waypoints[i][1]) for i in range(len(waypoints) - 1)]
    assert spacings[:2] == pytest.approx([1.0, 1.0], abs=1.0e-6)
