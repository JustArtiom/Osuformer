from __future__ import annotations

import math
from typing import List, Sequence

from .base import DifficultyHitObject, OsuStrainSkill
from .evaluators import AimEvaluator, RhythmEvaluator, SpeedEvaluator
from .strain_utils import count_top_weighted_sliders


class Aim(OsuStrainSkill):
    skill_multiplier: float = 25.6
    strain_decay_base: float = 0.15

    def __init__(self, mods: Sequence[str], include_sliders: bool) -> None:
        super().__init__(list(mods))
        self.include_sliders = include_sliders
        self._current_strain = 0.0
        self._slider_strains: List[float] = []

    def _strain_decay(self, ms: float) -> float:
        return math.pow(self.strain_decay_base, ms / 1000.0)

    def calculate_initial_strain(self, time: float, current: DifficultyHitObject) -> float:
        prev = current.previous(0)
        if prev is None:
            return 0.0
        return self._current_strain * self._strain_decay(time - prev.start_time)

    def strain_value_at(self, current: DifficultyHitObject) -> float:
        prev = current.previous(0)
        if prev is None:
            return 0.0

        delta_time = current.delta_time
        self._current_strain *= self._strain_decay(delta_time)
        difficulty = AimEvaluator.evaluate(current, self.include_sliders)
        self._current_strain += difficulty * self.skill_multiplier

        if current.base_object.object_type == "Slider":
            self._slider_strains.append(self._current_strain)

        return self._current_strain

    def get_difficult_sliders(self) -> float:
        if not self._slider_strains:
            return 0.0

        max_slider_strain = max(self._slider_strains)
        if max_slider_strain <= 0:
            return 0.0

        total = 0.0
        for strain in self._slider_strains:
            total += 1.0 / (1.0 + math.exp(-(strain / max_slider_strain * 12.0 - 6.0)))
        return total

    def count_top_weighted_sliders(self) -> float:
        return count_top_weighted_sliders(self._slider_strains, self.difficulty_value())


class Speed(OsuStrainSkill):
    skill_multiplier: float = 1.46
    strain_decay_base: float = 0.3
    reduced_section_count: int = 5

    def __init__(self, mods: Sequence[str]) -> None:
        super().__init__(list(mods))
        self._current_strain = 0.0
        self._current_rhythm = 0.0
        self._mods = list(mods)

    def _strain_decay(self, ms: float) -> float:
        return math.pow(self.strain_decay_base, ms / 1000.0)

    def calculate_initial_strain(self, time: float, current: DifficultyHitObject) -> float:
        prev = current.previous(0)
        if prev is None:
            return 0.0
        return (self._current_strain * self._current_rhythm) * self._strain_decay(time - prev.start_time)

    def strain_value_at(self, current: DifficultyHitObject) -> float:
        prev = current.previous(0)
        if prev is None:
            return 0.0

        strain_time = getattr(current, "strain_time", current.delta_time)
        self._current_strain *= self._strain_decay(strain_time)
        difficulty = SpeedEvaluator.evaluate(current, self._mods)
        self._current_strain += difficulty * self.skill_multiplier

        self._current_rhythm = RhythmEvaluator.evaluate(current)
        total_strain = self._current_strain * self._current_rhythm

        return total_strain

    def relevant_note_count(self) -> float:
        if not self.object_strains:
            return 0.0
        max_strain = max(self.object_strains)
        if max_strain <= 0:
            return 0.0
        return sum(1.0 / (1.0 + math.exp(-(strain / max_strain * 12.0 - 6.0))) for strain in self.object_strains)

    def count_top_weighted_sliders(self) -> float:
        return 0.0


class Flashlight(OsuStrainSkill):
    skill_multiplier: float = 0.05512
    strain_decay_base: float = 0.15

    def __init__(self, mods: Sequence[str]) -> None:
        super().__init__(list(mods))
        self._current_strain = 0.0
        self._has_hidden = any(mod.lower() == "hidden" for mod in mods)

    def _strain_decay(self, ms: float) -> float:
        return math.pow(self.strain_decay_base, ms / 1000.0)

    def calculate_initial_strain(self, time: float, current: DifficultyHitObject) -> float:
        prev = current.previous(0)
        if prev is None:
            return 0.0
        return self._current_strain * self._strain_decay(time - prev.start_time)

    def strain_value_at(self, current: DifficultyHitObject) -> float:
        prev = current.previous(0)
        if prev is None:
            return 0.0
        self._current_strain *= self._strain_decay(current.delta_time)
        return self._current_strain

    def difficulty_value(self) -> float:
        return sum(self.get_current_strain_peaks())

    @staticmethod
    def difficulty_to_performance(difficulty: float) -> float:
        return 25.0 * math.pow(difficulty, 2)
