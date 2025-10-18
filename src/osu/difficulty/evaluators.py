from __future__ import annotations

import math
from typing import Iterable, Sequence

from .math_utils import (
    clamp,
    reverse_lerp,
    smoothstep,
    smootherstep,
    smoothstep_bell_curve,
    logistic,
    logistic_simple,
    milliseconds_to_bpm,
    bpm_to_milliseconds,
)
from .preprocessing import OsuDifficultyHitObject


def _has_mod(mods: Sequence[str], *names: str) -> bool:
    mod_names = {mod.lower() for mod in mods}
    return any(name.lower() in mod_names for name in names)


class AimEvaluator:
    WIDE_ANGLE_MULTIPLIER = 1.5
    ACUTE_ANGLE_MULTIPLIER = 2.55
    SLIDER_MULTIPLIER = 1.35
    VELOCITY_CHANGE_MULTIPLIER = 0.75
    WIGGLE_MULTIPLIER = 1.02

    @classmethod
    def evaluate(cls, current: OsuDifficultyHitObject, include_sliders: bool) -> float:
        if current.base_object.object_type == "Spinner" or current.index <= 1:
            return 0.0

        prev = current.previous(0)
        prev_prev = current.previous(1)
        prev_prev_prev = current.previous(2)

        if prev is None or prev.base_object.object_type == "Spinner":
            return 0.0
        if prev_prev is None:
            return 0.0

        radius = OsuDifficultyHitObject.NORMALISED_RADIUS
        diameter = OsuDifficultyHitObject.NORMALISED_DIAMETER

        curr_velocity = current.lazy_jump_distance / current.adjusted_delta_time
        if prev.base_object.object_type == "Slider" and include_sliders:
            travel_velocity = prev.travel_distance / max(prev.travel_time, OsuDifficultyHitObject.MIN_DELTA_TIME)
            movement_velocity = current.minimum_jump_distance / max(current.minimum_jump_time, OsuDifficultyHitObject.MIN_DELTA_TIME)
            curr_velocity = max(curr_velocity, movement_velocity + travel_velocity)

        prev_velocity = prev.lazy_jump_distance / prev.adjusted_delta_time
        if prev_prev.base_object.object_type == "Slider" and include_sliders:
            travel_velocity = prev_prev.travel_distance / max(prev_prev.travel_time, OsuDifficultyHitObject.MIN_DELTA_TIME)
            movement_velocity = prev.minimum_jump_distance / max(prev.minimum_jump_time, OsuDifficultyHitObject.MIN_DELTA_TIME)
            prev_velocity = max(prev_velocity, movement_velocity + travel_velocity)

        wide_angle_bonus = 0.0
        acute_angle_bonus = 0.0
        slider_bonus = 0.0
        velocity_change_bonus = 0.0
        wiggle_bonus = 0.0

        aim_strain = curr_velocity

        if current.angle is not None and prev.angle is not None:
            curr_angle = current.angle
            last_angle = prev.angle

            angle_bonus = min(curr_velocity, prev_velocity)

            if max(current.adjusted_delta_time, prev.adjusted_delta_time) < 1.25 * min(current.adjusted_delta_time, prev.adjusted_delta_time):
                acute_angle_bonus = cls._calc_acute_angle_bonus(curr_angle)
                acute_angle_bonus *= 0.08 + 0.92 * (1 - min(acute_angle_bonus, math.pow(cls._calc_acute_angle_bonus(last_angle), 3)))
                acute_angle_bonus *= (
                    angle_bonus
                    * smootherstep(milliseconds_to_bpm(current.adjusted_delta_time, 2), 300, 400)
                    * smootherstep(current.lazy_jump_distance, diameter, diameter * 2)
                )

            wide_angle_bonus = cls._calc_wide_angle_bonus(curr_angle)
            wide_angle_bonus *= 1 - min(wide_angle_bonus, math.pow(cls._calc_wide_angle_bonus(last_angle), 3))
            wide_angle_bonus *= angle_bonus * smootherstep(current.lazy_jump_distance, 0, diameter)

            wiggle_bonus = (
                angle_bonus
                * smootherstep(current.lazy_jump_distance, radius, diameter)
                * math.pow(reverse_lerp(current.lazy_jump_distance, diameter * 3, diameter), 1.8)
                * smootherstep(curr_angle, math.radians(110), math.radians(60))
                * smootherstep(prev.lazy_jump_distance, radius, diameter)
                * math.pow(reverse_lerp(prev.lazy_jump_distance, diameter * 3, diameter), 1.8)
                * smootherstep(prev.angle or 0, math.radians(110), math.radians(60))
            )

            if prev_prev_prev is not None:
                last_base_pos = prev.base_object.position
                last_prev_pos = prev_prev.base_object.position

                distance = math.hypot(last_prev_pos[0] - last_base_pos[0], last_prev_pos[1] - last_base_pos[1])
                if distance < 1:
                    wide_angle_bonus *= 1 - 0.35 * (1 - distance)

        if max(prev_velocity, curr_velocity) != 0:
            prev_velocity = (prev.lazy_jump_distance + prev_prev.travel_distance) / prev.adjusted_delta_time
            curr_velocity = (current.lazy_jump_distance + prev.travel_distance) / current.adjusted_delta_time

            dist_ratio = smoothstep(abs(prev_velocity - curr_velocity) / max(prev_velocity, curr_velocity), 0, 1)
            overlap_velocity_buff = min(
                diameter * 1.25 / min(current.adjusted_delta_time, prev.adjusted_delta_time),
                abs(prev_velocity - curr_velocity),
            )
            velocity_change_bonus = overlap_velocity_buff * dist_ratio
            velocity_change_bonus *= math.pow(
                min(current.adjusted_delta_time, prev.adjusted_delta_time) / max(current.adjusted_delta_time, prev.adjusted_delta_time),
                2,
            )

        if prev.base_object.object_type == "Slider":
            slider_bonus = prev.travel_distance / max(prev.travel_time, OsuDifficultyHitObject.MIN_DELTA_TIME)

        aim_strain += wiggle_bonus * cls.WIGGLE_MULTIPLIER
        aim_strain += velocity_change_bonus * cls.VELOCITY_CHANGE_MULTIPLIER
        aim_strain += max(acute_angle_bonus * cls.ACUTE_ANGLE_MULTIPLIER, wide_angle_bonus * cls.WIDE_ANGLE_MULTIPLIER)
        aim_strain *= current.small_circle_bonus

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
    MIN_SPEED_BONUS = 200
    SPEED_BALANCING_FACTOR = 40
    DISTANCE_MULTIPLIER = 0.8

    @classmethod
    def evaluate(cls, current: OsuDifficultyHitObject, mods: Sequence[str]) -> float:
        if current.base_object.object_type == "Spinner":
            return 0.0

        prev = current.previous(0)
        if prev is None:
            return 0.0

        strain_time = current.adjusted_delta_time
        doubletapness = 1.0 - current.get_doubletapness(current.next(0))  # type: ignore[arg-type]

        strain_time /= clamp((strain_time / current.hit_window_great) / 0.93, 0.92, 1.0)

        speed_bonus = 0.0
        if milliseconds_to_bpm(strain_time) > cls.MIN_SPEED_BONUS:
            speed_bonus = 0.75 * math.pow((bpm_to_milliseconds(cls.MIN_SPEED_BONUS) - strain_time) / cls.SPEED_BALANCING_FACTOR, 2)

        travel_distance = prev.travel_distance
        distance = travel_distance + current.minimum_jump_distance
        distance = min(distance, cls.SINGLE_SPACING_THRESHOLD)
        distance_bonus = math.pow(distance / cls.SINGLE_SPACING_THRESHOLD, 3.95) * cls.DISTANCE_MULTIPLIER
        distance_bonus *= math.sqrt(current.small_circle_bonus)

        if _has_mod(mods, "AutoPilot"):
            distance_bonus = 0.0

        difficulty = (1.0 + speed_bonus + distance_bonus) * 1000.0 / strain_time
        return difficulty * doubletapness


class RhythmEvaluator:
    HISTORY_TIME_MAX = 5 * 1000
    HISTORY_OBJECTS_MAX = 32
    RHYTHM_OVERALL_MULTIPLIER = 1.0
    RHYTHM_RATIO_MULTIPLIER = 15.0

    @classmethod
    def evaluate(cls, current: OsuDifficultyHitObject) -> float:
        if current.base_object.object_type == "Spinner":
            return 0.0

        rhythm_complexity_sum = 0.0
        delta_difference_epsilon = current.hit_window_great * 0.3

        island = _Island(delta_difference_epsilon)
        previous_island = _Island(delta_difference_epsilon)
        island_counts: list[tuple[_Island, int]] = []

        start_ratio = 0.0
        first_delta_switch = False

        historical_note_count = min(current.index, cls.HISTORY_OBJECTS_MAX)
        rhythm_start = 0

        while rhythm_start < historical_note_count - 2:
            prev_candidate = current.previous(rhythm_start)
            if prev_candidate is None:
                break
            if current.start_time - prev_candidate.start_time >= cls.HISTORY_TIME_MAX:
                break
            rhythm_start += 1

        prev = current.previous(rhythm_start)
        last = current.previous(rhythm_start + 1)
        if prev is None or last is None:
            return 1.0

        for i in range(rhythm_start, 0, -1):
            curr = current.previous(i - 1)
            if curr is None:
                break

            time_decay = (cls.HISTORY_TIME_MAX - (current.start_time - curr.start_time)) / cls.HISTORY_TIME_MAX
            note_decay = (historical_note_count - i) / historical_note_count if historical_note_count else 0.0
            curr_historical_decay = min(note_decay, time_decay)

            curr_delta = max(curr.delta_time, 1e-7)
            prev_delta = max(prev.delta_time, 1e-7)
            last_delta = max(last.delta_time, 1e-7)

            delta_difference = max(prev_delta, curr_delta) / min(prev_delta, curr_delta)
            delta_difference_fraction = delta_difference - math.trunc(delta_difference)
            curr_ratio = 1.0 + cls.RHYTHM_RATIO_MULTIPLIER * min(0.5, smoothstep_bell_curve(delta_difference_fraction))
            difference_multiplier = clamp(2.0 - delta_difference / 8.0, 0.0, 1.0)
            window_penalty = min(1.0, max(0.0, abs(prev_delta - curr_delta) - delta_difference_epsilon) / delta_difference_epsilon)

            effective_ratio = window_penalty * curr_ratio * difference_multiplier

            if first_delta_switch:
                if abs(prev_delta - curr_delta) < delta_difference_epsilon:
                    island.add_delta(int(curr_delta))
                else:
                    if curr.base_object.object_type == "Slider":
                        effective_ratio *= 0.125
                    if prev.base_object.object_type == "Slider":
                        effective_ratio *= 0.3
                    if island.is_similar_polarity(previous_island):
                        effective_ratio *= 0.5
                    if last_delta > prev_delta + delta_difference_epsilon and prev_delta > curr_delta + delta_difference_epsilon:
                        effective_ratio *= 0.125
                    if previous_island.delta_count == island.delta_count:
                        effective_ratio *= 0.5

                    found_index = next(
                        (idx for idx, (isle, _) in enumerate(island_counts) if isle == island),
                        None,
                    )
                    if found_index is not None:
                        isle, count = island_counts[found_index]
                        if previous_island == island:
                            count += 1
                        power = logistic(island.delta, midpoint_offset=58.33, multiplier=0.24, max_value=2.75)
                        effective_ratio *= min(3.0 / count, math.pow(1.0 / count, power))
                        island_counts[found_index] = (isle, count)
                    else:
                        island_counts.append((island.copy(), 1))

                    doubletapness = prev.get_doubletapness(curr if isinstance(curr, OsuDifficultyHitObject) else None)
                    effective_ratio *= 1 - doubletapness * 0.75

                    rhythm_complexity_sum += math.sqrt(effective_ratio * start_ratio) * curr_historical_decay
                    start_ratio = effective_ratio
                    previous_island = island.copy()

                    if prev_delta + delta_difference_epsilon < curr_delta:
                        first_delta_switch = False

                    island = _Island(int(curr_delta), delta_difference_epsilon)

            elif prev_delta > curr_delta + delta_difference_epsilon:
                first_delta_switch = True

                if curr.base_object.object_type == "Slider":
                    effective_ratio *= 0.6
                if prev.base_object.object_type == "Slider":
                    effective_ratio *= 0.6

                start_ratio = effective_ratio
                island = _Island(int(curr_delta), delta_difference_epsilon)

            last = prev
            prev = curr

        rhythm_difficulty = math.sqrt(4 + rhythm_complexity_sum * cls.RHYTHM_OVERALL_MULTIPLIER) / 2.0
        rhythm_difficulty *= 1 - current.get_doubletapness(current.next(0))  # type: ignore[arg-type]

        return rhythm_difficulty


class _Island:
    def __init__(self, epsilon_or_delta: float, epsilon: float | None = None) -> None:
        if epsilon is None:
            self.delta_difference_epsilon = epsilon_or_delta
            self.delta = math.inf
            self.delta_count = 0
        else:
            self.delta_difference_epsilon = epsilon
            self.delta = max(epsilon_or_delta, OsuDifficultyHitObject.MIN_DELTA_TIME)
            self.delta_count = 1

    def copy(self) -> "_Island":
        new = _Island(self.delta_difference_epsilon)
        new.delta = self.delta
        new.delta_count = self.delta_count
        return new

    def add_delta(self, delta: int) -> None:
        if math.isinf(self.delta):
            self.delta = max(delta, OsuDifficultyHitObject.MIN_DELTA_TIME)
        self.delta_count += 1

    def is_similar_polarity(self, other: "_Island") -> bool:
        return self.delta_count % 2 == other.delta_count % 2

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Island):
            return False
        return (
            abs(self.delta - other.delta) < self.delta_difference_epsilon
            and self.delta_count == other.delta_count
        )
