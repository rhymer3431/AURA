from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from systems.perception.observation import PerceptionObservationService
from systems.shared.contracts.observation import RawObservation


def test_perception_normalizes_raw_observation_without_viewer() -> None:
    service = PerceptionObservationService(viewer_publisher=None)
    raw = RawObservation(
        rgb=np.full((2, 3, 3), 0.5, dtype=np.float32),
        depth=np.asarray([[1.0, np.nan, np.inf]], dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
        camera_pos_w=np.asarray((1.0, 2.0, 3.0), dtype=np.float32),
        camera_rot_w=np.eye(3, dtype=np.float32),
        robot_state=SimpleNamespace(base_pos_w=np.asarray((0.0, 0.0, 0.8), dtype=np.float32), base_yaw=0.25),
        stamp_s=12.5,
    )

    frame = service.ingest(raw)

    assert frame.rgb.dtype == np.uint8
    assert frame.rgb.shape == (2, 3, 3)
    assert int(frame.rgb[0, 0, 0]) == 127
    assert frame.depth.dtype == np.float32
    assert frame.depth.tolist() == [[1.0, 0.0, 0.0]]
    health = service.latest_health()
    assert health["status"] == "running"
    assert health["last_error"] is None
