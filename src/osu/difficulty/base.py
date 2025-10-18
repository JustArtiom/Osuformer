from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .math_utils import clamp


@dataclass
class DifficultyHitObject:
    base_object: "DifficultyObject"
    last_object: "DifficultyObject"
    clock_rate: float
    objects: List["DifficultyHitObject"]
    index: int

    delta_time: float = field(init=False)
    start_time: float = field(init=False)
    end_time: float = field(init=False)

    def __post_init__(self) -> None:
        self.delta_time = (self.base_object.start_time - self.last_object.start_time) / self.clock_rate
        self.start_time = self.base_object.start_time / self.clock_rate
        self.end_time = self.base_object.end_time / self.clock_rate

    def previous(self, backwards_index: int) -> Optional["DifficultyHitObject"]:
        idx = self.index - (backwards_index + 1)
        if 0 <= idx < len(self.objects):
            return self.objects[idx]
        return None

    def next(self, forwards_index: int) -> Optional["DifficultyHitObject"]:
        idx = self.index + (forwards_index + 1)
        if 0 <= idx < len(self.objects):
            return self.objects[idx]
        return None


class Skill:
    def __init__(self, mods: List[str]) -> None:
        self._mods = tuple(mods)

    def process(self, current: DifficultyHitObject) -> None:
        raise NotImplementedError

    def difficulty_value(self) -> float:
        raise NotImplementedError


class StrainSkill(Skill):
    decay_weight: float = 0.9
    section_length: int = 400

    def __init__(self, mods: List[str]) -> None:
        super().__init__(mods)
        self._current_section_peak = 0.0
        self._current_section_end = 0.0
        self._strain_peaks: List[float] = []
        self.object_strains: List[float] = []

    def strain_value_at(self, current: DifficultyHitObject) -> float:
        raise NotImplementedError

    def calculate_initial_strain(self, time: float, current: DifficultyHitObject) -> float:
        raise NotImplementedError

    def process(self, current: DifficultyHitObject) -> None:
        if current.index == 0:
            self._current_section_end = math.ceil(current.start_time / self.section_length) * self.section_length

        while current.start_time > self._current_section_end:
            self._save_current_peak()
            self._start_new_section_from(self._current_section_end, current)
            self._current_section_end += self.section_length

        strain = self.strain_value_at(current)
        self._current_section_peak = max(strain, self._current_section_peak)
        self.object_strains.append(strain)

    def count_top_weighted_strains(self) -> float:
        if not self.object_strains:
            return 0.0

        consistent_top_strain = self.difficulty_value() / 10.0
        if consistent_top_strain == 0:
            return len(self.object_strains)

        return sum(1.1 / (1.0 + math.exp(-10.0 * (s / consistent_top_strain - 0.88))) for s in self.object_strains)

    def _save_current_peak(self) -> None:
        self._strain_peaks.append(self._current_section_peak)

    def _start_new_section_from(self, time: float, current: DifficultyHitObject) -> None:
        self._current_section_peak = self.calculate_initial_strain(time, current)

    def get_current_strain_peaks(self) -> List[float]:
        return [*self._strain_peaks, self._current_section_peak]

    def difficulty_value(self) -> float:
        peaks = [p for p in self.get_current_strain_peaks() if p > 0]
        peaks.sort(reverse=True)

        difficulty = 0.0
        weight = 1.0
        for strain in peaks:
            difficulty += strain * weight
            weight *= self.decay_weight
        return difficulty


class OsuStrainSkill(StrainSkill):
    reduced_section_count: int = 10
    reduced_strain_baseline: float = 0.75

    def difficulty_value(self) -> float:
        peaks = [p for p in self.get_current_strain_peaks() if p > 0]
        peaks.sort(reverse=True)

        limited = peaks[:]
        for i in range(min(len(limited), self.reduced_section_count)):
            scale = math.log10(
                clamp(i / self.reduced_section_count, 0.0, 1.0) * 9.0 + 1.0
            )
            limited[i] *= self.reduced_strain_baseline + (1.0 - self.reduced_strain_baseline) * scale

        limited.sort(reverse=True)

        difficulty = 0.0
        weight = 1.0
        for strain in limited:
            difficulty += strain * weight
            weight *= self.decay_weight
        return difficulty

    @staticmethod
    def difficulty_to_performance(difficulty: float) -> float:
        return math.pow(5.0 * max(1.0, difficulty / 0.0675) - 4.0, 3.0) / 100000.0


@dataclass
class DifficultyObject:
    start_time: float
    end_time: float
    position: tuple[float, float]
    end_position: tuple[float, float]
    object_radius: float
    object_type: str
    slider_length: float = 0.0
    slider_duration: float = 0.0
    slider_repeat_count: int = 0
    nested_count: int = 0
    hit_window_great: float = 0.0

