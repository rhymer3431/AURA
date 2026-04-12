from __future__ import annotations

import numpy as np

from backend.webrtc.tracks import _render_depth_preview_image


def test_depth_preview_handles_nonfinite_input_without_warnings_or_wraparound() -> None:
    depth = np.array(
        [
            [np.nan, np.inf, -np.inf],
            [-1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    preview = _render_depth_preview_image(depth)

    assert preview.shape == (2, 3, 3)
    assert preview.dtype == np.uint8
    assert np.array_equal(preview[0, 0], np.array([0, 0, 0], dtype=np.uint8))
    assert np.array_equal(preview[0, 1], np.array([0, 0, 0], dtype=np.uint8))
    assert np.array_equal(preview[1, 2], np.array([255, 255, 255], dtype=np.uint8))
