from __future__ import annotations

import math


DIFFICULTY_MULTIPLIER = 0.0675
PERFORMANCE_BASE_MULTIPLIER = 1.15


def calculate_difficulty_rating(difficulty_value: float) -> float:
    return math.sqrt(max(difficulty_value, 0.0)) * DIFFICULTY_MULTIPLIER


def calculate_star_rating_from_performance(base_performance: float) -> float:
    if base_performance <= 1e-5:
        return 0.0
    return math.pow(PERFORMANCE_BASE_MULTIPLIER, 1.0 / 3.0) * 0.027 * (
        math.pow(100000.0 / math.pow(2.0, 1.0 / 1.1) * base_performance, 1.0 / 3.0) + 4.0
    )


def difficulty_to_performance(difficulty: float) -> float:
    return math.pow(5.0 * max(1.0, difficulty / DIFFICULTY_MULTIPLIER) - 4.0, 3.0) / 100000.0
