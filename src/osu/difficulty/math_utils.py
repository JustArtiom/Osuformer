from __future__ import annotations

import math
from typing import Iterable


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def reverse_lerp(x: float, start: float, end: float) -> float:
    if math.isclose(start, end):
        return 0.0
    return clamp((x - start) / (end - start), 0.0, 1.0)


def smoothstep(x: float, start: float, end: float) -> float:
    if math.isclose(start, end):
        return 0.0
    t = clamp((x - start) / (end - start), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(x: float, start: float, end: float) -> float:
    if math.isclose(start, end):
        return 0.0
    t = clamp((x - start) / (end - start), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def smoothstep_bell_curve(x: float, mean: float = 0.5, width: float = 0.5) -> float:
    offset = x - mean
    if offset > 0:
        offset = width - offset
    else:
        offset = width + offset
    return smoothstep(offset, 0.0, width)


def logistic(
    x: float,
    midpoint_offset: float,
    multiplier: float,
    max_value: float = 1.0,
) -> float:
    return max_value / (1.0 + math.exp(multiplier * (midpoint_offset - x)))


def logistic_simple(exponent: float, max_value: float = 1.0) -> float:
    return max_value / (1.0 + math.exp(exponent))


def norm(p: float, values: Iterable[float]) -> float:
    return math.pow(sum(math.pow(x, p) for x in values), 1.0 / p)


def bpm_to_milliseconds(bpm: float, delimiter: int = 4) -> float:
    return 60000.0 / delimiter / bpm


def milliseconds_to_bpm(ms: float, delimiter: int = 4) -> float:
    return 60000.0 / (ms * delimiter)


def erf(x: float) -> float:
    if x == 0:
        return 0.0
    if math.isinf(x):
        return 1.0 if x > 0 else -1.0

    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    tau = t * (
        0.254829592
        + t
        * (
            -0.284496736
            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))
        )
    )
    result = 1.0 - tau * math.exp(-x * x)
    return result if x >= 0 else -result


def erfc(x: float) -> float:
    return 1.0 - erf(x)


def erf_inv(x: float) -> float:
    if x <= -1:
        return -math.inf
    if x >= 1:
        return math.inf
    if x == 0:
        return 0.0

    a = 0.147
    sgn = math.copysign(1.0, x)
    abs_x = abs(x)
    ln = math.log(1 - abs_x * abs_x)
    t1 = 2 / (math.pi * a) + ln / 2
    t2 = ln / a
    base = math.sqrt(t1 * t1 - t2) - t1

    correction = 0.0
    if abs_x >= 0.85:
        correction = math.pow((abs_x - 0.85) / 0.293, 8)

    return sgn * (math.sqrt(base) + correction)


def erfc_inv(x: float) -> float:
    return erf_inv(1.0 - x)

