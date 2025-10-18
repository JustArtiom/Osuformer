from __future__ import annotations

import math
from typing import Iterable, Sequence, Set

from .math_utils import reverse_lerp


DIFFICULTY_MULTIPLIER = 0.0675


def calculate_difficulty_rating(difficulty_value: float) -> float:
    return math.sqrt(max(difficulty_value, 0.0)) * DIFFICULTY_MULTIPLIER


class OsuRatingCalculator:
    def __init__(
        self,
        mods: Sequence[str],
        total_hits: int,
        approach_rate: float,
        overall_difficulty: float,
        mechanical_difficulty_rating: float,
        slider_factor: float,
    ) -> None:
        self.mods: Set[str] = {m.lower() for m in mods}
        self.total_hits = total_hits
        self.approach_rate = approach_rate
        self.overall_difficulty = overall_difficulty
        self.mechanical_difficulty_rating = mechanical_difficulty_rating
        self.slider_factor = slider_factor

    def compute_aim_rating(self, aim_difficulty_value: float) -> float:
        if "autopilot" in self.mods:
            return 0.0

        aim_rating = calculate_difficulty_rating(aim_difficulty_value)

        if "touchdevice" in self.mods:
            aim_rating = math.pow(aim_rating, 0.8)
        if "relax" in self.mods:
            aim_rating *= 0.9

        rating_multiplier = self._approach_rate_bonus(relax="relax" in self.mods)

        if "hidden" in self.mods:
            visibility_factor = calculate_aim_visibility_factor(self.approach_rate, self.mechanical_difficulty_rating)
            rating_multiplier += calculate_visibility_bonus(
                self.mods,
                self.approach_rate,
                visibility_factor=visibility_factor,
                slider_factor=self.slider_factor,
            )

        rating_multiplier *= 0.98 + math.pow(max(0.0, self.overall_difficulty), 2) / 2500.0

        return aim_rating * math.pow(max(rating_multiplier, 1e-6), 1.0 / 3.0)

    def compute_speed_rating(self, speed_difficulty_value: float) -> float:
        if "relax" in self.mods:
            return 0.0

        speed_rating = calculate_difficulty_rating(speed_difficulty_value)

        if "autopilot" in self.mods:
            speed_rating *= 0.5

        rating_multiplier = self._approach_rate_bonus(autopilot="autopilot" in self.mods)

        if "hidden" in self.mods:
            visibility_factor = calculate_speed_visibility_factor(self.approach_rate, self.mechanical_difficulty_rating)
            rating_multiplier += calculate_visibility_bonus(self.mods, self.approach_rate, visibility_factor=visibility_factor)

        rating_multiplier *= 0.95 + math.pow(max(0.0, self.overall_difficulty), 2) / 750.0

        return speed_rating * math.pow(max(rating_multiplier, 1e-6), 1.0 / 3.0)

    def compute_flashlight_rating(self, flashlight_difficulty_value: float) -> float:
        if "flashlight" not in self.mods:
            return 0.0
        flashlight_rating = calculate_difficulty_rating(flashlight_difficulty_value)
        if "touchdevice" in self.mods:
            flashlight_rating = math.pow(flashlight_rating, 0.8)
        if "relax" in self.mods:
            flashlight_rating *= 0.7
        elif "autopilot" in self.mods:
            flashlight_rating *= 0.4
        rating_multiplier = (
            0.7
            + 0.1 * min(1.0, self.total_hits / 200.0)
            + (0.2 * min(1.0, (self.total_hits - 200) / 200.0) if self.total_hits > 200 else 0.0)
        )
        rating_multiplier *= 0.98 + math.pow(max(0.0, self.overall_difficulty), 2) / 2500.0
        return flashlight_rating * math.sqrt(max(rating_multiplier, 1e-6))

    def _approach_rate_bonus(self, *, relax: bool = False, autopilot: bool = False) -> float:
        approach_rate_length_bonus = (
            0.95 + 0.4 * min(1.0, self.total_hits / 2000.0) + (math.log10(self.total_hits / 2000.0) * 0.5 if self.total_hits > 2000 else 0.0)
        )
        approach_rate_factor = 0.0
        if self.approach_rate > 10.33:
            approach_rate_factor = 0.3 * (self.approach_rate - 10.33)
        elif self.approach_rate < 8.0 and not relax:
            approach_rate_factor = 0.05 * (8.0 - self.approach_rate)
        if autopilot:
            approach_rate_factor = 0.0
        return 1.0 + approach_rate_factor * approach_rate_length_bonus


def calculate_mechanical_difficulty_rating(aim_difficulty_value: float, speed_difficulty_value: float) -> float:
    aim_value = difficulty_to_performance(calculate_difficulty_rating(aim_difficulty_value))
    speed_value = difficulty_to_performance(calculate_difficulty_rating(speed_difficulty_value))
    total_value = math.pow(math.pow(aim_value, 1.1) + math.pow(speed_value, 1.1), 1 / 1.1)
    return calculate_star_rating_from_performance(total_value)


def calculate_star_rating_from_performance(base_performance: float) -> float:
    if base_performance <= 1e-5:
        return 0.0
    return math.pow(1.14, 1.0 / 3.0) * 0.0265 * (
        math.pow(100000 / math.pow(2.0, 1 / 1.1) * base_performance, 1.0 / 3.0) + 4.0
    )


def difficulty_to_performance(difficulty: float) -> float:
    return math.pow(5.0 * max(1.0, difficulty / 0.0675) - 4.0, 3.0) / 100000.0


def calculate_aim_visibility_factor(approach_rate: float, mechanical_difficulty_rating: float) -> float:
    ar_factor_end_point = 11.5
    mechanical_factor = reverse_lerp(mechanical_difficulty_rating, 5, 10)
    ar_factor_start = 9 + (10.33 - 9) * mechanical_factor
    return reverse_lerp(approach_rate, ar_factor_end_point, ar_factor_start)


def calculate_speed_visibility_factor(approach_rate: float, mechanical_difficulty_rating: float) -> float:
    ar_factor_end_point = 11.5
    mechanical_factor = reverse_lerp(mechanical_difficulty_rating, 5, 10)
    ar_factor_start = 10 + (10.33 - 10) * mechanical_factor
    return reverse_lerp(approach_rate, ar_factor_end_point, ar_factor_start)


def calculate_visibility_bonus(
    mods: Iterable[str],
    approach_rate: float,
    *,
    visibility_factor: float = 1.0,
    slider_factor: float = 1.0,
) -> float:
    mod_set = {m.lower() for m in mods}
    if "hidden" not in mod_set:
        return 0.0

    is_partially_visible = False
    reading_bonus = (0.025 if is_partially_visible else 0.04) * (12.0 - max(approach_rate, 7.0))
    reading_bonus *= visibility_factor

    slider_visibility_factor = math.pow(slider_factor, 3.0)

    if approach_rate < 7.0:
        reading_bonus += (0.02 if is_partially_visible else 0.045) * (7.0 - max(approach_rate, 0.0)) * slider_visibility_factor
    if approach_rate < 0.0:
        reading_bonus += (0.01 if is_partially_visible else 0.1) * (1 - math.pow(1.5, approach_rate)) * slider_visibility_factor

    return max(0.0, reading_bonus)
