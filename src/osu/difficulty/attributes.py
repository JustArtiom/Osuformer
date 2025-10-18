from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class DifficultyAttributes:
    star_rating: float
    aim: float
    speed: float
    aim_difficulty_value: float
    speed_difficulty_value: float
    flashlight_rating: float
    approach_rate: float
    overall_difficulty: float
    circle_size: float
    clock_rate: float
    max_combo: int
    hit_circle_count: int
    slider_count: int
    spinner_count: int
    strains: Sequence[float]
    mods: Sequence[str]
    slider_factor: float
    aim_difficult_slider_count: float
    speed_note_count: float


@dataclass(slots=True)
class PerformanceAttributes:
    pp: float
    aim_pp: float
    speed_pp: float
    accuracy_pp: float
    accuracy: float
    effective_miss_count: float
