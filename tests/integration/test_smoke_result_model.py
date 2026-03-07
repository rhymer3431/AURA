from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime.smoke_result_model import aggregate_smoke_result


def test_smoke_result_aggregation_allows_empty_detection_pipeline_pass() -> None:
    result = aggregate_smoke_result(
        target_tier="memory",
        frame_received=True,
        rgb_ready=True,
        depth_ready=False,
        pose_ready=True,
        detection_attempted=True,
        detections_nonempty=False,
        perception_ingress_ready=True,
        memory_ingress_ready=True,
        memory_update_called=False,
    )

    assert result.sensor_status.passed is True
    assert result.pipeline_status.passed is True
    assert result.memory_status.status == "memory_empty_batch"
    assert result.memory_status.passed is False
    assert result.empty_observation_batch is True
    assert result.overall_status == "pipeline_smoke_pass"
