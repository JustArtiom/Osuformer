import math

from src.osu_style_classifier.features.geometry import (
    distance,
    entropy_bits,
    flow_angle_deg,
    mean,
    stddev,
)


def test_distance_pythagorean() -> None:
    assert distance((0.0, 0.0), (3.0, 4.0)) == 5.0


def test_flow_angle_straight_line_is_zero() -> None:
    assert flow_angle_deg((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)) == 0.0


def test_flow_angle_reversal_is_180() -> None:
    assert math.isclose(flow_angle_deg((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)), 180.0, abs_tol=1e-6)


def test_mean_stddev_entropy() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert math.isclose(stddev([1.0, 2.0, 3.0]), math.sqrt(2 / 3), abs_tol=1e-9)
    assert math.isclose(entropy_bits([1, 1]), 1.0, abs_tol=1e-9)
