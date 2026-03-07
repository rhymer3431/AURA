from __future__ import annotations

import pytest

from common.geometry import within_xy_radius, xy_distance


def test_xy_distance_uses_planar_distance_only():
    distance = xy_distance([0.0, 0.0, 0.8], [0.3, 0.4, 3.0])

    assert distance == pytest.approx(0.5)


def test_within_xy_radius_is_true_on_boundary():
    assert within_xy_radius([0.0, 0.0, 0.8], [0.8, 0.0, 0.0], 0.8) is True


def test_within_xy_radius_is_false_past_boundary():
    assert within_xy_radius([0.0, 0.0, 0.8], [0.81, 0.0, 0.0], 0.8) is False
