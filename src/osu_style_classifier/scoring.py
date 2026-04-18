from __future__ import annotations

import math


def soft_above(value: float, threshold: float, width: float) -> float:
    if width <= 0:
        return 1.0 if value >= threshold else 0.0
    return 1.0 / (1.0 + math.exp(-(value - threshold) / width))


def soft_below(value: float, threshold: float, width: float) -> float:
    if width <= 0:
        return 1.0 if value <= threshold else 0.0
    return 1.0 / (1.0 + math.exp((value - threshold) / width))


def soft_between(value: float, low: float, high: float, width: float) -> float:
    return soft_above(value, low, width) * soft_below(value, high, width)


def soft_and(*scores: float) -> float:
    if not scores:
        return 0.0
    return min(scores)


def soft_or(*scores: float) -> float:
    if not scores:
        return 0.0
    return max(scores)
