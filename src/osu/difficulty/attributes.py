from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class DifficultyAttributes:
    star_rating: float
    aim_difficulty: float
    speed_difficulty: float
    flashlight_difficulty: float
    slider_factor: float
    aim_difficult_slider_count: float
    speed_note_count: float
    aim_difficult_strain_count: float
    speed_difficult_strain_count: float
    approach_rate: float
    overall_difficulty: float
    drain_rate: float
    circle_size: float
    clock_rate: float
    max_combo: int
    hit_circle_count: int
    slider_count: int
    spinner_count: int
    mods: Sequence[str]
    strains: Sequence[float]

    @property
    def aim(self) -> float:
        return self.aim_difficulty

    @property
    def speed(self) -> float:
        return self.speed_difficulty

    @property
    def flashlight_rating(self) -> float:
        return self.flashlight_difficulty


@dataclass
class PerformanceAttributes:
    pp: float
    aim_pp: float
    speed_pp: float
    accuracy_pp: float
    accuracy: float
    effective_miss_count: float
