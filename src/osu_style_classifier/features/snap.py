from __future__ import annotations

from src.osu.timing_point import TimingPoint


COMMON_DIVISORS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)


def active_uninherited(timing_points: list[TimingPoint], time: float) -> TimingPoint | None:
    last: TimingPoint | None = None
    for tp in timing_points:
        if tp.uninherited != 1:
            continue
        if tp.time <= time:
            last = tp
        else:
            break
    if last is None:
        for tp in timing_points:
            if tp.uninherited == 1:
                return tp
    return last


def active_inherited_multiplier(timing_points: list[TimingPoint], time: float) -> float:
    multiplier = 1.0
    for tp in timing_points:
        if tp.time > time:
            break
        if tp.uninherited == 0 and tp.beat_length < 0:
            multiplier = -100.0 / tp.beat_length
        elif tp.uninherited == 1:
            multiplier = 1.0
    return multiplier


def classify_snap(delta_ms: float, beat_length_ms: float, tolerance_ms: float = 8.0) -> int | None:
    if beat_length_ms <= 0 or delta_ms <= 0:
        return None
    best: tuple[int, float] | None = None
    for d in COMMON_DIVISORS:
        expected = beat_length_ms / d
        err = abs(delta_ms - expected)
        if err <= tolerance_ms:
            if best is None or d > best[0]:
                best = (d, err)
    if best is not None:
        return best[0]
    ratio = beat_length_ms / delta_ms
    rounded = max(1, round(ratio))
    expected = beat_length_ms / rounded
    if abs(delta_ms - expected) <= tolerance_ms * 1.5:
        return rounded
    return None
