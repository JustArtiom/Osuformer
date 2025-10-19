from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .base import DifficultyHitObject, DifficultyObject


def _vec_sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]

def _vec_mul(a: tuple[float, float], scalar: float) -> tuple[float, float]:
    return a[0] * scalar, a[1] * scalar


def _vec_length(v: tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _vec_dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _vec_det(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


@dataclass
class OsuDifficultyHitObject(DifficultyHitObject):
    NORMALISED_RADIUS: int = 50
    NORMALISED_DIAMETER: int = NORMALISED_RADIUS * 2
    MIN_DELTA_TIME: int = 25
    MAXIMUM_SLIDER_RADIUS: float = NORMALISED_RADIUS * 2.4
    ASSUMED_SLIDER_RADIUS: float = NORMALISED_RADIUS * 1.8

    strain_time: float = 0.0
    lazy_jump_distance: float = 0.0
    minimum_jump_distance: float = 0.0
    minimum_jump_time: float = 0.0
    travel_distance: float = 0.0
    travel_time: float = 0.0
    lazy_travel_distance: float = 0.0
    lazy_travel_time: float = 0.0
    lazy_end_position: Optional[tuple[float, float]] = None
    angle: Optional[float] = None
    hit_window_great: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()

        base = self.base_object
        last = self.previous(0)
        last_last = self.previous(1)

        self.strain_time = max(self.delta_time, self.MIN_DELTA_TIME)
        self.hit_window_great = (2.0 * base.hit_window_great) / self.clock_rate if base.hit_window_great else 0.0

        self._initialise_slider_values(base)
        self._set_distances(last)

        if last_last is not None and last_last.base_object.object_type != "Spinner":
            last_cursor = self._get_end_cursor_position(last)
            last_last_cursor = self._get_end_cursor_position(last_last)

            v1 = _vec_sub(last_last_cursor, last.base_object.stacked_position)
            v2 = _vec_sub(base.stacked_position, last_cursor)

            dot = _vec_dot(v1, v2)
            det = _vec_det(v1, v2)
            if not math.isclose(v1[0], 0.0) or not math.isclose(v1[1], 0.0):
                self.angle = abs(math.atan2(det, dot))

    def _initialise_slider_values(self, base: DifficultyObject) -> None:
        if base.object_type != "Slider":
            self.travel_distance = 0.0
            self.travel_time = 0.0
            self.lazy_travel_distance = 0.0
            self.lazy_travel_time = 0.0
            self.lazy_end_position = base.stacked_end_position
            return

        repeat_bonus = math.pow(1.0 + max(base.slider_repeat_count - 1, 0) / 2.5, 1.0 / 2.5)

        self.lazy_travel_distance = base.lazy_travel_distance
        self.lazy_travel_time = base.lazy_travel_time
        self.travel_distance = base.lazy_travel_distance * repeat_bonus
        self.travel_time = max(self.lazy_travel_time / self.clock_rate, self.MIN_DELTA_TIME)
        self.lazy_end_position = base.stacked_end_position

    def _set_distances(self, last: Optional[OsuDifficultyHitObject]) -> None:
        base = self.base_object

        if base.object_type == "Slider":
            if self.travel_distance == 0.0:
                self.travel_distance = self.lazy_travel_distance
            if self.travel_time == 0.0:
                self.travel_time = max(self.lazy_travel_time / self.clock_rate, self.MIN_DELTA_TIME)

        if base.object_type == "Spinner" or (last and last.base_object.object_type == "Spinner"):
            self.lazy_jump_distance = 0.0
            self.minimum_jump_distance = 0.0
            self.minimum_jump_time = self.strain_time
            return

        scaling_factor = self.NORMALISED_RADIUS / base.object_radius if base.object_radius > 0 else 1.0
        if base.object_radius < 30.0:
            small_circle_bonus = min(30.0 - base.object_radius, 5.0) / 50.0
            scaling_factor *= 1.0 + small_circle_bonus

        last_cursor = self._get_end_cursor_position(last) if last else base.stacked_position

        self.lazy_jump_distance = _vec_length(_vec_mul(_vec_sub(base.stacked_position, last_cursor), scaling_factor))
        self.minimum_jump_time = self.strain_time
        self.minimum_jump_distance = self.lazy_jump_distance

        if last and last.base_object.object_type == "Slider":
            last_travel_time = max(last.lazy_travel_time / self.clock_rate, self.MIN_DELTA_TIME)
            self.minimum_jump_time = max(self.strain_time - last_travel_time, self.MIN_DELTA_TIME)
            tail_jump_vector = _vec_sub(last.base_object.stacked_end_position, base.stacked_position)
            tail_jump_distance = _vec_length(_vec_mul(tail_jump_vector, scaling_factor))
            self.minimum_jump_distance = max(
                0.0,
                min(
                    self.lazy_jump_distance - (self.MAXIMUM_SLIDER_RADIUS - self.ASSUMED_SLIDER_RADIUS),
                    tail_jump_distance - self.MAXIMUM_SLIDER_RADIUS,
                ),
            )

        self.travel_distance = max(self.travel_distance, 0.0)
        self.minimum_jump_distance = max(self.minimum_jump_distance, 0.0)

    def _get_end_cursor_position(self, difficulty_hit_object: Optional["OsuDifficultyHitObject"]) -> tuple[float, float]:
        if difficulty_hit_object is None:
            return self.base_object.stacked_position
        return difficulty_hit_object.lazy_end_position or difficulty_hit_object.base_object.stacked_end_position

    def get_doubletapness(self, osu_next_obj: Optional["OsuDifficultyHitObject"]) -> float:
        if osu_next_obj is None:
            return 0.0

        curr_delta_time = max(1.0, self.delta_time)
        next_delta_time = max(1.0, osu_next_obj.delta_time)
        delta_difference = abs(next_delta_time - curr_delta_time)
        speed_ratio = curr_delta_time / max(curr_delta_time, delta_difference)
        window_ratio = math.pow(min(1.0, curr_delta_time / self.hit_window_great), 2.0) if self.hit_window_great > 0 else 0.0
        return 1.0 - math.pow(speed_ratio, 1.0 - window_ratio)
