from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from .math_utils import (
    clamp,
    reverse_lerp,
    smoothstep,
    smootherstep,
    milliseconds_to_bpm,
    bpm_to_milliseconds,
    logistic,
)
from .preprocessing import OsuDifficultyHitObject


def _has_mod(mods: Sequence[str], *names: str) -> bool:
    mod_names = {mod.lower() for mod in mods}
    return any(name.lower() in mod_names for name in names)


class AimEvaluator:
    WIDE_ANGLE_MULTIPLIER = 1.5
    ACUTE_ANGLE_MULTIPLIER = 2.6
    SLIDER_MULTIPLIER = 1.35
    VELOCITY_CHANGE_MULTIPLIER = 0.75
    WIGGLE_MULTIPLIER = 1.02

    @classmethod
    def evaluate(cls, current: OsuDifficultyHitObject, include_sliders: bool) -> float:
        if (
            current.base_object.object_type == "Spinner"
            or current.index <= 1
            or current.previous(0).base_object.object_type == "Spinner"
        ):
            return 0.0

        prev = current.previous(0)
        prev_prev = current.previous(1)
        if prev is None or prev_prev is None:
            return 0.0

        radius = OsuDifficultyHitObject.NORMALISED_RADIUS
        diameter = OsuDifficultyHitObject.NORMALISED_DIAMETER

        curr_velocity = current.lazy_jump_distance / current.strain_time if current.strain_time > 0 else 0.0
        if include_sliders and prev.base_object.object_type == "Slider":
            travel_velocity = prev.travel_distance / prev.travel_time if prev.travel_time > 0 else 0.0
            movement_velocity = current.minimum_jump_distance / current.minimum_jump_time if current.minimum_jump_time > 0 else 0.0
            curr_velocity = max(curr_velocity, movement_velocity + travel_velocity)

        prev_velocity = prev.lazy_jump_distance / prev.strain_time if prev.strain_time > 0 else 0.0
        if include_sliders and prev_prev.base_object.object_type == "Slider":
            travel_velocity = prev_prev.travel_distance / prev_prev.travel_time if prev_prev.travel_time > 0 else 0.0
            movement_velocity = prev.minimum_jump_distance / prev.minimum_jump_time if prev.minimum_jump_time > 0 else 0.0
            prev_velocity = max(prev_velocity, movement_velocity + travel_velocity)

        wide_angle_bonus = 0.0
        acute_angle_bonus = 0.0
        slider_bonus = 0.0
        velocity_change_bonus = 0.0
        wiggle_bonus = 0.0

        aim_strain = curr_velocity

        if max(current.strain_time, prev.strain_time) < 1.25 * min(current.strain_time, prev.strain_time):
            if current.angle is not None and prev.angle is not None:
                curr_angle = current.angle
                last_angle = prev.angle

                angle_bonus = min(curr_velocity, prev_velocity)

                wide_angle_bonus = cls._calc_wide_angle_bonus(curr_angle)
                acute_angle_bonus = cls._calc_acute_angle_bonus(curr_angle)

                wide_angle_bonus *= 1 - min(wide_angle_bonus, math.pow(cls._calc_wide_angle_bonus(last_angle), 3))
                acute_angle_bonus *= 0.08 + 0.92 * (1 - min(acute_angle_bonus, math.pow(cls._calc_acute_angle_bonus(last_angle), 3)))

                wide_angle_bonus *= angle_bonus * smootherstep(current.lazy_jump_distance, 0.0, diameter)
                acute_angle_bonus *= (
                    angle_bonus
                    * smootherstep(milliseconds_to_bpm(current.strain_time, 2), 300, 400)
                    * smootherstep(current.lazy_jump_distance, diameter, diameter * 2)
                )

                wiggle_bonus = (
                    angle_bonus
                    * smootherstep(current.lazy_jump_distance, radius, diameter)
                    * math.pow(reverse_lerp(current.lazy_jump_distance, diameter * 3, diameter), 1.8)
                    * smootherstep(curr_angle, math.radians(110), math.radians(60))
                    * smootherstep(prev.lazy_jump_distance, radius, diameter)
                    * math.pow(reverse_lerp(prev.lazy_jump_distance, diameter * 3, diameter), 1.8)
                    * smootherstep(last_angle, math.radians(110), math.radians(60))
                )

        if max(prev_velocity, curr_velocity) > 0:
            prev_velocity = (prev.lazy_jump_distance + prev_prev.travel_distance) / prev.strain_time if prev.strain_time > 0 else 0.0
            curr_velocity = (current.lazy_jump_distance + prev.travel_distance) / current.strain_time if current.strain_time > 0 else 0.0

            max_velocity = max(prev_velocity, curr_velocity)
            if max_velocity > 0:
                dist_ratio = math.pow(math.sin(math.pi / 2 * abs(prev_velocity - curr_velocity) / max_velocity), 2)
                min_strain = min(current.strain_time, prev.strain_time)
                max_strain = max(current.strain_time, prev.strain_time)
                overlap_velocity_buff = min(diameter * 1.25 / min_strain if min_strain > 0 else 0.0, abs(prev_velocity - curr_velocity))
                velocity_change_bonus = overlap_velocity_buff * dist_ratio
                if max_strain > 0:
                    velocity_change_bonus *= math.pow(min_strain / max_strain, 2)

        if include_sliders and prev.base_object.object_type == "Slider":
            slider_bonus = prev.travel_distance / prev.travel_time if prev.travel_time > 0 else 0.0

        aim_strain += wiggle_bonus * cls.WIGGLE_MULTIPLIER
        aim_strain += max(
            acute_angle_bonus * cls.ACUTE_ANGLE_MULTIPLIER,
            wide_angle_bonus * cls.WIDE_ANGLE_MULTIPLIER + velocity_change_bonus * cls.VELOCITY_CHANGE_MULTIPLIER,
        )

        if include_sliders:
            aim_strain += slider_bonus * cls.SLIDER_MULTIPLIER

        return aim_strain

    @staticmethod
    def _calc_wide_angle_bonus(angle: float) -> float:
        return smoothstep(angle, math.radians(40), math.radians(140))

    @staticmethod
    def _calc_acute_angle_bonus(angle: float) -> float:
        return smoothstep(angle, math.radians(140), math.radians(40))


class SpeedEvaluator:
    SINGLE_SPACING_THRESHOLD = OsuDifficultyHitObject.NORMALISED_DIAMETER * 1.25
    MIN_SPEED_BONUS = 200.0
    SPEED_BALANCING_FACTOR = 40.0
    DISTANCE_MULTIPLIER = 0.9

    @classmethod
    def evaluate(cls, current: OsuDifficultyHitObject, mods: Sequence[str]) -> float:
        if current.base_object.object_type == "Spinner":
            return 0.0

        prev = current.previous(0)
        if prev is None:
            return 0.0

        strain_time = current.strain_time
        if current.hit_window_great > 0:
            strain_time /= clamp((strain_time / current.hit_window_great) / 0.93, 0.92, 1.0)

        doubletapness = 1.0 - current.get_doubletapness(current.next(0))  # type: ignore[arg-type]

        speed_bonus = 0.0
        if milliseconds_to_bpm(strain_time) > cls.MIN_SPEED_BONUS:
            speed_bonus = 0.75 * math.pow((bpm_to_milliseconds(cls.MIN_SPEED_BONUS) - strain_time) / cls.SPEED_BALANCING_FACTOR, 2)

        travel_distance = prev.travel_distance
        distance = min(travel_distance + current.minimum_jump_distance, cls.SINGLE_SPACING_THRESHOLD)
        distance_bonus = math.pow(distance / cls.SINGLE_SPACING_THRESHOLD, 3.95) * cls.DISTANCE_MULTIPLIER

        if _has_mod(mods, "autopilot"):
            distance_bonus = 0.0

        difficulty = (1.0 + speed_bonus + distance_bonus) * 1000.0 / strain_time if strain_time > 0 else 0.0
        return difficulty * doubletapness


class RhythmEvaluator:
    HISTORY_TIME_MAX = 5 * 1000
    HISTORY_OBJECTS_MAX = 32
    RHYTHM_OVERALL_MULTIPLIER = 0.95
    RHYTHM_RATIO_MULTIPLIER = 12.0

    @classmethod
    def evaluate(cls, current: OsuDifficultyHitObject) -> float:
        if current.base_object.object_type == "Spinner":
            return 0.0

        delta_difference_epsilon = current.hit_window_great * 0.3
        island = _Island(delta_difference_epsilon)
        previous_island = _Island(delta_difference_epsilon)
        island_counts: list[tuple[_Island, int]] = []

        rhythm_complexity_sum = 0.0
        start_ratio = 0.0
        first_delta_switch = False

        historical_note_count = min(current.index, cls.HISTORY_OBJECTS_MAX)

        rhythm_start = 0
        while rhythm_start < historical_note_count - 2:
            prev_candidate = current.previous(rhythm_start)
            if prev_candidate is None or current.start_time - prev_candidate.start_time >= cls.HISTORY_TIME_MAX:
                break
            rhythm_start += 1

        prev_obj = current.previous(rhythm_start)
        last_obj = current.previous(rhythm_start + 1)
        if prev_obj is None or last_obj is None:
            return 1.0

        prev_obj = prev_obj  # type: ignore[assignment]
        last_obj = last_obj  # type: ignore[assignment]

        for i in range(rhythm_start, 0, -1):
            curr_obj = current.previous(i - 1)
            if curr_obj is None:
                break

            time_decay = (cls.HISTORY_TIME_MAX - (current.start_time - curr_obj.start_time)) / cls.HISTORY_TIME_MAX
            note_decay = (historical_note_count - i) / historical_note_count if historical_note_count > 0 else 0.0
            curr_historical_decay = max(0.0, min(note_decay, time_decay))

            curr_delta = getattr(curr_obj, "strain_time", 0.0)
            prev_delta = getattr(prev_obj, "strain_time", 0.0)
            last_delta = getattr(last_obj, "strain_time", 0.0)

            if curr_delta <= 0 or prev_delta <= 0:
                prev_obj = curr_obj
                last_obj = prev_obj
                continue

            delta_difference_ratio = min(prev_delta, curr_delta) / max(prev_delta, curr_delta)
            if delta_difference_ratio <= 0:
                delta_difference_ratio = 1.0

            curr_ratio = 1.0 + cls.RHYTHM_RATIO_MULTIPLIER * min(0.5, math.pow(math.sin(math.pi / delta_difference_ratio), 2))

            fraction = max(prev_delta / curr_delta, curr_delta / prev_delta)
            fraction_multiplier = clamp(2.0 - fraction / 8.0, 0.0, 1.0)

            if delta_difference_epsilon <= 0:
                window_penalty = 1.0
            else:
                window_penalty = min(1.0, max(0.0, abs(prev_delta - curr_delta) - delta_difference_epsilon) / delta_difference_epsilon)

            effective_ratio = window_penalty * curr_ratio * fraction_multiplier

            if first_delta_switch:
                if abs(prev_delta - curr_delta) < delta_difference_epsilon:
                    island.add_delta(int(curr_delta))
                else:
                    if curr_obj.base_object.object_type == "Slider":
                        effective_ratio *= 0.125

                    if prev_obj.base_object.object_type == "Slider":
                        effective_ratio *= 0.3

                    if island.is_similar_polarity(previous_island):
                        effective_ratio *= 0.5

                    if last_delta > prev_delta + delta_difference_epsilon and prev_delta > curr_delta + delta_difference_epsilon:
                        effective_ratio *= 0.125

                    if previous_island.delta_count == island.delta_count:
                        effective_ratio *= 0.5

                    existing_index = _find_island_index(island_counts, island)
                    if existing_index is not None:
                        existing_island, existing_count = island_counts[existing_index]
                        if previous_island.equals(island):
                            existing_count += 1
                        power = logistic(island.delta or 0.0, midpoint_offset=58.33, multiplier=0.24, max_value=2.75)
                        effective_ratio *= min(3.0 / existing_count, math.pow(1.0 / existing_count, power))
                        island_counts[existing_index] = (existing_island, existing_count)
                    else:
                        island_counts.append((island, 1))

                    doubletapness = prev_obj.get_doubletapness(curr_obj)
                    effective_ratio *= 1 - doubletapness * 0.75

                    rhythm_complexity_sum += math.sqrt(max(0.0, effective_ratio * start_ratio)) * curr_historical_decay
                    start_ratio = effective_ratio

                    previous_island = island

                    if prev_delta + delta_difference_epsilon < curr_delta:
                        first_delta_switch = False

                    island = _Island(delta_difference_epsilon, int(curr_delta))
            elif prev_delta > curr_delta + delta_difference_epsilon:
                first_delta_switch = True

                if curr_obj.base_object.object_type == "Slider":
                    effective_ratio *= 0.6

                if prev_obj.base_object.object_type == "Slider":
                    effective_ratio *= 0.6

                start_ratio = effective_ratio
                island = _Island(delta_difference_epsilon, int(curr_delta))

            last_obj = prev_obj
            prev_obj = curr_obj

        return math.sqrt(4.0 + rhythm_complexity_sum * cls.RHYTHM_OVERALL_MULTIPLIER) / 2.0


class _Island:
    def __init__(self, epsilon: float, delta: Optional[int] = None) -> None:
        self._epsilon = epsilon
        self.delta: Optional[int] = None
        self.delta_count: int = 0
        if delta is not None:
            self.delta = max(delta, OsuDifficultyHitObject.MIN_DELTA_TIME)
            self.delta_count = 1

    def add_delta(self, delta: int) -> None:
        value = max(delta, OsuDifficultyHitObject.MIN_DELTA_TIME)
        if self.delta is None:
            self.delta = value
        self.delta_count += 1

    def is_similar_polarity(self, other: "_Island") -> bool:
        return self.delta_count % 2 == other.delta_count % 2

    def equals(self, other: Optional["_Island"]) -> bool:
        if other is None or self.delta is None or other.delta is None:
            return False
        return abs(self.delta - other.delta) < self._epsilon and self.delta_count == other.delta_count


def _find_island_index(islands: Iterable[tuple[_Island, int]], target: _Island) -> Optional[int]:
    for idx, (candidate, _) in enumerate(islands):
        if candidate.equals(target):
            return idx
    return None
