from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.depth_visualization import build_rgb_depth_panel, compute_depth_display_range, depth_to_heatmap_bgr, sanitize_depth_image


def test_sanitize_depth_image_marks_invalid_pixels() -> None:
    depth = np.asarray([[np.nan, -1.0, 0.0], [1.5, 7.2, np.inf]], dtype=np.float32)

    sanitized, valid = sanitize_depth_image(depth, 5.0)

    assert sanitized.shape == (2, 3)
    assert valid.tolist() == [[False, False, False], [True, True, False]]
    assert np.isclose(sanitized[1, 0], 1.5)
    assert np.isclose(sanitized[1, 1], 5.0)


def test_depth_to_heatmap_bgr_masks_invalid_pixels() -> None:
    depth = np.asarray([[0.0, 0.5], [2.5, 5.0]], dtype=np.float32)

    heatmap = depth_to_heatmap_bgr(depth, 5.0)

    assert heatmap.shape == (2, 2, 3)
    assert tuple(int(v) for v in heatmap[0, 0]) == (0, 0, 0)
    assert int(heatmap[0, 1, 2]) > int(heatmap[1, 1, 2])


def test_build_rgb_depth_panel_concatenates_rgb_and_depth_views() -> None:
    rgb = np.asarray(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    depth = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    panel = build_rgb_depth_panel(rgb, depth, 5.0)

    assert panel.shape == (2, 4, 3)
    assert tuple(int(v) for v in panel[0, 0]) == (0, 0, 255)
    assert np.any(panel[:, 2:, :] != 0)


def test_compute_depth_display_range_uses_valid_percentiles() -> None:
    depth = np.asarray([[0.0, 1.0, 2.0], [3.0, 8.0, 12.0]], dtype=np.float32)

    display_min, display_max = compute_depth_display_range(depth, default_max_m=5.0, min_percentile=10.0, max_percentile=90.0)

    assert 0.0 <= display_min < display_max
    assert display_max <= 12.0
