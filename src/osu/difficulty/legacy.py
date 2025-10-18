from __future__ import annotations

def calculate_scale_from_circle_size(circle_size: float, apply_fudge: bool = False) -> float:
    broken_gamefield_rounding_allowance = 1.00041 if apply_fudge else 1.0
    difficulty_range = (circle_size - 5.0) / 5.0
    scale = (1.0 - 0.7 * difficulty_range) / 2.0
    return float(scale * broken_gamefield_rounding_allowance)

