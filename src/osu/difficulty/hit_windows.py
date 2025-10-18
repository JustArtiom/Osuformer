from __future__ import annotations

from enum import Enum, auto

from .math_utils import clamp


class HitResult(Enum):
    Great = auto()
    Ok = auto()
    Meh = auto()
    Miss = auto()


class DifficultyRange:
    __slots__ = ("min", "mid", "max")

    def __init__(self, minimum: float, mid: float, maximum: float) -> None:
        self.min = minimum
        self.mid = mid
        self.max = maximum


def difficulty_range(difficulty: float, range_: DifficultyRange) -> float:
    if difficulty > 5.0:
        return range_.mid + (range_.max - range_.mid) * ((difficulty - 5.0) / 5.0)
    if difficulty < 5.0:
        return range_.mid + (range_.mid - range_.min) * ((difficulty - 5.0) / 5.0)
    return range_.mid


def inverse_difficulty_range(value: float, range_: DifficultyRange) -> float:
    if (value - range_.mid) == 0:
        return 5.0
    if value > range_.mid:
        return ((value - range_.mid) / (range_.max - range_.mid)) * 5.0 + 5.0
    return ((value - range_.mid) / (range_.mid - range_.min)) * 5.0 + 5.0


class HitWindows:
    def __init__(self) -> None:
        self._great = 0.0
        self._ok = 0.0
        self._meh = 0.0

    def set_difficulty(self, difficulty: float) -> None:
        raise NotImplementedError

    def window_for(self, result: HitResult) -> float:
        raise NotImplementedError


class OsuHitWindows(HitWindows):
    GREAT_WINDOW_RANGE = DifficultyRange(80, 50, 20)
    OK_WINDOW_RANGE = DifficultyRange(140, 100, 60)
    MEH_WINDOW_RANGE = DifficultyRange(200, 150, 100)
    MISS_WINDOW = 400.0

    def set_difficulty(self, difficulty: float) -> None:
        self._great = difficulty_range(difficulty, self.GREAT_WINDOW_RANGE) - 0.5
        self._ok = difficulty_range(difficulty, self.OK_WINDOW_RANGE) - 0.5
        self._meh = difficulty_range(difficulty, self.MEH_WINDOW_RANGE) - 0.5

    def window_for(self, result: HitResult) -> float:
        if result is HitResult.Great:
            return self._great
        if result is HitResult.Ok:
            return self._ok
        if result is HitResult.Meh:
            return self._meh
        if result is HitResult.Miss:
            return self.MISS_WINDOW
        raise ValueError(f"Unsupported HitResult: {result}")
