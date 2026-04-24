from __future__ import annotations

from fractions import Fraction

DURATION_FRACTIONS: tuple[tuple[int, int], ...] = (
    (1, 16),
    (1, 12),
    (1, 8),
    (1, 6),
    (3, 16),
    (1, 4),
    (1, 3),
    (3, 8),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 1),
    (5, 4),
    (4, 3),
    (3, 2),
    (5, 3),
    (7, 4),
    (2, 1),
    (5, 2),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (8, 1),
    (12, 1),
    (16, 1),
    (24, 1),
    (32, 1),
)

DURATION_COUNT = len(DURATION_FRACTIONS)

_FRACTION_FLOATS: tuple[float, ...] = tuple(n / d for n, d in DURATION_FRACTIONS)


def beats_to_duration_index(beats: float) -> int:
    if beats <= 0:
        return 0
    best_idx = 0
    best_err = abs(beats - _FRACTION_FLOATS[0])
    for i in range(1, DURATION_COUNT):
        err = abs(beats - _FRACTION_FLOATS[i])
        if err < best_err:
            best_err = err
            best_idx = i
    return best_idx


def duration_index_to_beats(index: int) -> float:
    if index < 0 or index >= DURATION_COUNT:
        raise ValueError(f"duration index {index} out of range [0, {DURATION_COUNT})")
    return _FRACTION_FLOATS[index]


def duration_index_to_fraction(index: int) -> Fraction:
    if index < 0 or index >= DURATION_COUNT:
        raise ValueError(f"duration index {index} out of range [0, {DURATION_COUNT})")
    num, den = DURATION_FRACTIONS[index]
    return Fraction(num, den)
