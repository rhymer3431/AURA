"""Helpers for inferring standalone policy observation layouts."""

from __future__ import annotations

from dataclasses import dataclass

from systems.world_state.domain.constants import (
    LEGACY_HEIGHT_SCAN_RESOLUTION,
    LEGACY_HEIGHT_SCAN_SIZE,
    TUNED_HEIGHT_SCAN_RESOLUTION,
    TUNED_HEIGHT_SCAN_SIZE,
)


BASE_OBSERVATION_DIM = 12
FEET_CONTACT_OBSERVATION_DIM = 2


@dataclass(frozen=True, slots=True)
class PolicyObservationLayout:
    """Observation layout inferred from a policy input shape."""

    input_obs_dim: int | None
    base_obs_dim: int
    height_scan_points: int | None
    feet_contact_dim: int


def height_scan_point_count(size: tuple[float, float], resolution: float) -> int:
    x_count = int(round(float(size[0]) / float(resolution))) + 1
    y_count = int(round(float(size[1]) / float(resolution))) + 1
    return x_count * y_count


def infer_policy_observation_layout(input_obs_dim: int | None, dof_count: int) -> PolicyObservationLayout:
    """Infer height-scan and feet-contact dimensions from a flat policy input size."""

    base_obs_dim = BASE_OBSERVATION_DIM + (3 * int(dof_count))
    if input_obs_dim is None:
        return PolicyObservationLayout(
            input_obs_dim=None,
            base_obs_dim=base_obs_dim,
            height_scan_points=None,
            feet_contact_dim=0,
        )

    residual_dim = int(input_obs_dim) - base_obs_dim
    if residual_dim < 0:
        raise RuntimeError(
            f"Invalid policy input observation dim {input_obs_dim}: base dim {base_obs_dim} exceeds input size."
        )

    legacy_scan_points = height_scan_point_count(LEGACY_HEIGHT_SCAN_SIZE, LEGACY_HEIGHT_SCAN_RESOLUTION)
    tuned_scan_points = height_scan_point_count(TUNED_HEIGHT_SCAN_SIZE, TUNED_HEIGHT_SCAN_RESOLUTION)
    supported_layouts = (
        (0, 0),
        (0, FEET_CONTACT_OBSERVATION_DIM),
        (legacy_scan_points, 0),
        (legacy_scan_points, FEET_CONTACT_OBSERVATION_DIM),
        (tuned_scan_points, 0),
        (tuned_scan_points, FEET_CONTACT_OBSERVATION_DIM),
    )
    for height_scan_points, feet_contact_dim in supported_layouts:
        if residual_dim == height_scan_points + feet_contact_dim:
            return PolicyObservationLayout(
                input_obs_dim=int(input_obs_dim),
                base_obs_dim=base_obs_dim,
                height_scan_points=height_scan_points,
                feet_contact_dim=feet_contact_dim,
            )

    raise RuntimeError(
        "Unsupported policy observation layout: "
        f"input dim {input_obs_dim} does not match the supported flat/scan/contact combinations "
        f"for {dof_count} DOFs."
    )


__all__ = [
    "BASE_OBSERVATION_DIM",
    "FEET_CONTACT_OBSERVATION_DIM",
    "PolicyObservationLayout",
    "height_scan_point_count",
    "infer_policy_observation_layout",
]
