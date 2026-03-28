from __future__ import annotations

from typing import Collection

from .math_utils import logistic


def count_top_weighted_sliders(slider_strains: Collection[float], difficulty_value: float) -> float:
    if not slider_strains:
        return 0.0

    consistent_top_strain = difficulty_value / 10.0
    if consistent_top_strain == 0:
        return 0.0

    return sum(
        logistic(strain / consistent_top_strain, midpoint_offset=0.88, multiplier=10.0, max_value=1.1)
        for strain in slider_strains
    )

