from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ..beatmap import Beatmap
from ..hit_object import Circle, Slider, Spinner
from ..mods import Mods
from .attributes import DifficultyAttributes, PerformanceAttributes
from .base import DifficultyObject
from .hit_windows import HitResult, OsuHitWindows, difficulty_range, inverse_difficulty_range, DifficultyRange
from .legacy import calculate_scale_from_circle_size
from .math_utils import clamp
from .mods import clock_rate_for_mods, normalise_mods
from .preprocessing import OsuDifficultyHitObject
from .rating import (
    OsuRatingCalculator,
    calculate_difficulty_rating,
    calculate_mechanical_difficulty_rating,
    calculate_star_rating_from_performance,
    difficulty_to_performance,
)
from .skills import Aim, Speed, Flashlight


PREEMPT_MAX = 1800.0
PREEMPT_MID = 1200.0
PREEMPT_MIN = 450.0


def calculate_rate_adjusted_approach_rate(approach_rate: float, clock_rate: float) -> float:
    preempt = difficulty_range(approach_rate, DifficultyRange(PREEMPT_MAX, PREEMPT_MID, PREEMPT_MIN)) / clock_rate
    return inverse_difficulty_range(preempt, DifficultyRange(PREEMPT_MAX, PREEMPT_MID, PREEMPT_MIN))


def calculate_rate_adjusted_overall_difficulty(overall_difficulty: float, clock_rate: float) -> float:
    hit_windows = OsuHitWindows()
    hit_windows.set_difficulty(overall_difficulty)
    hit_window_great = hit_windows.window_for(HitResult.Great) / clock_rate
    return (79.5 - hit_window_great) / 6.0


def calculate_difficulty(beatmap: Beatmap, mods: Sequence[Mods | str] | None = None) -> DifficultyAttributes:
    mods_list = normalise_mods(mods)
    clock_rate = clock_rate_for_mods(mods_list)

    difficulty = beatmap.difficulty
    approach_rate = float(difficulty.approach_rate)
    overall_difficulty = float(difficulty.overall_difficulty)
    circle_size = float(difficulty.circle_size)

    if "HardRock" in mods_list:
        approach_rate = min(10.0, approach_rate * 1.4)
        overall_difficulty = min(10.0, overall_difficulty * 1.4)
        circle_size = min(10.0, circle_size * 1.3)
    if "Easy" in mods_list:
        approach_rate *= 0.5
        overall_difficulty *= 0.5
        circle_size *= 0.5

    approach_rate_rate_adjusted = calculate_rate_adjusted_approach_rate(approach_rate, clock_rate)
    overall_difficulty_rate_adjusted = calculate_rate_adjusted_overall_difficulty(overall_difficulty, clock_rate)

    hit_windows = OsuHitWindows()
    hit_windows.set_difficulty(overall_difficulty)
    hit_window_great = hit_windows.window_for(HitResult.Great)
    per_object_hit_window = hit_window_great

    radius = 64.0 * calculate_scale_from_circle_size(circle_size, apply_fudge=True)

    difficulty_objects = _generate_difficulty_objects(
        beatmap,
        radius,
        per_object_hit_window,
        approach_rate=approach_rate,
        stack_leniency=getattr(getattr(beatmap, "general", None), "stack_leniency", 0.7),
    )

    if len(difficulty_objects) <= 1:
        return DifficultyAttributes(
            star_rating=0.0,
            aim=0.0,
            speed=0.0,
            aim_rating=0.0,
            speed_rating=0.0,
            flashlight_rating=0.0,
            approach_rate=approach_rate_rate_adjusted,
            overall_difficulty=overall_difficulty_rate_adjusted,
            circle_size=circle_size,
            clock_rate=clock_rate,
            max_combo=len(beatmap.hit_objects),
            hit_circle_count=sum(isinstance(obj, Circle) for obj in beatmap.hit_objects),
            slider_count=sum(isinstance(obj, Slider) for obj in beatmap.hit_objects),
            spinner_count=sum(isinstance(obj, Spinner) for obj in beatmap.hit_objects),
            strains=[],
            mods=mods_list,
            slider_factor=1.0,
            aim_difficult_slider_count=0.0,
            speed_note_count=0.0,
        )

    difficulty_hit_objects: List[OsuDifficultyHitObject] = []
    for idx in range(1, len(difficulty_objects)):
        current = difficulty_objects[idx]
        last = difficulty_objects[idx - 1]
        diff_obj = OsuDifficultyHitObject(
            base_object=current,
            last_object=last,
            clock_rate=clock_rate,
            objects=difficulty_hit_objects,
            index=len(difficulty_hit_objects),
        )
        difficulty_hit_objects.append(diff_obj)

    aim_skill = Aim(mods_list, include_sliders=True)
    aim_no_sliders_skill = Aim(mods_list, include_sliders=False)
    speed_skill = Speed(mods_list)
    flashlight_skill = None
    if any(mod.lower() == "flashlight" for mod in mods_list):
        flashlight_skill = Flashlight(mods_list, has_hidden=any(mod.lower() == "hidden" for mod in mods_list))

    for diff_obj in difficulty_hit_objects:
        aim_skill.process(diff_obj)
        aim_no_sliders_skill.process(diff_obj)
        speed_skill.process(diff_obj)
        if flashlight_skill is not None:
            flashlight_skill.process(diff_obj)

    aim_difficulty_value = aim_skill.difficulty_value()
    aim_no_slider_difficulty_value = aim_no_sliders_skill.difficulty_value()
    speed_difficulty_value = speed_skill.difficulty_value()
    flashlight_difficulty_value = flashlight_skill.difficulty_value() if flashlight_skill is not None else 0.0

    difficult_sliders = aim_skill.get_difficult_sliders()
    speed_notes = speed_skill.relevant_note_count()

    if aim_difficulty_value == 0.0:
        slider_factor = 1.0
    else:
        slider_factor = (
            calculate_difficulty_rating(aim_no_slider_difficulty_value) / calculate_difficulty_rating(aim_difficulty_value)
            if aim_no_slider_difficulty_value > 0
            else 1.0
        )

    mechanical_difficulty_rating = calculate_mechanical_difficulty_rating(aim_difficulty_value, speed_difficulty_value)

    rating_calculator = OsuRatingCalculator(
        mods_list,
        total_hits=len(difficulty_objects),
        approach_rate=approach_rate_rate_adjusted,
        overall_difficulty=overall_difficulty_rate_adjusted,
        mechanical_difficulty_rating=mechanical_difficulty_rating,
        slider_factor=slider_factor,
    )

    aim_rating = rating_calculator.compute_aim_rating(aim_difficulty_value)
    speed_rating = rating_calculator.compute_speed_rating(speed_difficulty_value)
    flashlight_rating = rating_calculator.compute_flashlight_rating(flashlight_difficulty_value)

    base_aim_performance = difficulty_to_performance(aim_rating)
    base_speed_performance = difficulty_to_performance(speed_rating)
    base_flashlight_performance = difficulty_to_performance(flashlight_rating)

    base_performance = math.pow(
        math.pow(base_aim_performance, 1.1)
        + math.pow(base_speed_performance, 1.1)
        + math.pow(base_flashlight_performance, 1.1),
        1.0 / 1.1,
    )

    star_rating = calculate_star_rating_from_performance(base_performance)

    hit_circle_count = sum(isinstance(obj, Circle) for obj in beatmap.hit_objects)
    slider_count = sum(isinstance(obj, Slider) for obj in beatmap.hit_objects)
    spinner_count = sum(isinstance(obj, Spinner) for obj in beatmap.hit_objects)

    return DifficultyAttributes(
        star_rating=star_rating,
        aim=aim_rating,
        speed=speed_rating,
        aim_difficulty_value=aim_difficulty_value,
        speed_difficulty_value=speed_difficulty_value,
        flashlight_rating=flashlight_rating,
        approach_rate=approach_rate_rate_adjusted,
        overall_difficulty=overall_difficulty_rate_adjusted,
        circle_size=circle_size,
        clock_rate=clock_rate,
        max_combo=len(beatmap.hit_objects),
        hit_circle_count=hit_circle_count,
        slider_count=slider_count,
        spinner_count=spinner_count,
        strains=list(aim_skill.object_strains),
        mods=mods_list,
        slider_factor=slider_factor,
        aim_difficult_slider_count=difficult_sliders,
        speed_note_count=speed_notes,
    )


def _generate_difficulty_objects(
    beatmap: Beatmap,
    radius: float,
    hit_window_great: float,
    *,
    approach_rate: float,
    stack_leniency: float,
) -> List[DifficultyObject]:
    objects: List[DifficultyObject] = []

    sorted_objects = sorted(beatmap.hit_objects, key=lambda obj: obj.time)
    stack_offsets = _compute_stack_offsets(sorted_objects, radius, approach_rate, stack_leniency)

    for idx, ho in enumerate(sorted_objects):
        stack_offset = stack_offsets[idx]
        if isinstance(ho, Circle):
            pos = _apply_stack_offset((float(ho.x), float(ho.y)), stack_offset)
            start_time = float(ho.time)
            objects.append(
                DifficultyObject(
                    start_time=start_time,
                    end_time=start_time,
                    position=pos,
                    end_position=pos,
                    object_radius=radius,
                    object_type="Circle",
                    hit_window_great=hit_window_great,
                )
            )
        elif isinstance(ho, Slider):
            pos = _apply_stack_offset((float(ho.x), float(ho.y)), stack_offset)
            start_time = float(ho.time)
            duration = float(getattr(ho.object_params, "duration", 0.0) or 0.0)
            end_time = start_time + duration
            end_pos = _apply_stack_offset(_get_slider_end_position(ho), stack_offset)
            slider_length = float(getattr(ho.object_params, "length", 0.0) or 0.0)
            repeat_count = int(getattr(ho.object_params, "slides", 1) or 1)
            objects.append(
                DifficultyObject(
                    start_time=start_time,
                    end_time=end_time,
                    position=pos,
                    end_position=end_pos,
                    object_radius=radius,
                    object_type="Slider",
                    slider_length=slider_length,
                    slider_duration=duration,
                    slider_repeat_count=repeat_count,
                    hit_window_great=hit_window_great,
                )
            )
        elif isinstance(ho, Spinner):
            pos = _apply_stack_offset((float(ho.x), float(ho.y)), stack_offset)
            start_time = float(ho.time)
            end_time = float(getattr(ho.object_params, "end_time", ho.time))
            objects.append(
                DifficultyObject(
                    start_time=start_time,
                    end_time=end_time,
                    position=pos,
                    end_position=pos,
                    object_radius=radius,
                    object_type="Spinner",
                    hit_window_great=hit_window_great,
                )
            )

    return objects


def _compute_stack_offsets(
    hit_objects: Sequence[Circle | Slider | Spinner],
    radius: float,
    approach_rate: float,
    stack_leniency: float,
) -> List[float]:
    if not hit_objects:
        return []

    scale = radius / 64.0 if radius > 0 else 1.0
    stack_distance = 3.0
    stack_threshold = difficulty_range(approach_rate, DifficultyRange(PREEMPT_MAX, PREEMPT_MID, PREEMPT_MIN)) * stack_leniency
    stack_threshold = max(stack_threshold, 0.0)

    stack_heights: List[int] = [0 for _ in hit_objects]

    for i, base in enumerate(hit_objects):
        if isinstance(base, Spinner):
            continue

        base_start = float(base.time)
        base_end = _get_end_time(base)
        base_pos = (float(base.x), float(base.y))
        base_tail_pos = _get_slider_end_position(base) if isinstance(base, Slider) else None
        current_end_time = base_end if base_end >= base_start else base_start

        for j in range(i + 1, len(hit_objects)):
            other = hit_objects[j]
            if isinstance(other, Spinner):
                continue

            other_start = float(other.time)
            if other_start - stack_threshold > current_end_time:
                break

            other_pos = (float(other.x), float(other.y))
            stacked = False

            if _distance(base_pos, other_pos) < stack_distance:
                stacked = True
            elif base_tail_pos is not None and _distance(base_tail_pos, other_pos) < stack_distance:
                stacked = True

            if stacked:
                stack_heights[i] += 1
                current_end_time = other_start

    offset_per_stack = -6.4 * scale
    return [height * offset_per_stack for height in stack_heights]


def _apply_stack_offset(position: Tuple[float, float], offset: float) -> Tuple[float, float]:
    if offset == 0.0:
        return position
    return position[0] + offset, position[1] + offset


def _get_slider_end_position(hit_object: Circle | Slider | Spinner) -> Tuple[float, float]:
    if not isinstance(hit_object, Slider):
        return float(hit_object.x), float(hit_object.y)

    curve_points = getattr(hit_object.object_params, "curve_points", []) or []
    if curve_points:
        tail_x, tail_y = curve_points[-1]
        return float(tail_x), float(tail_y)
    return float(hit_object.x), float(hit_object.y)


def _get_end_time(hit_object: Circle | Slider | Spinner) -> float:
    if isinstance(hit_object, Slider):
        duration = float(getattr(hit_object.object_params, "duration", 0.0) or 0.0)
        return float(hit_object.time) + duration
    if isinstance(hit_object, Spinner):
        return float(getattr(hit_object.object_params, "end_time", hit_object.time))
    return float(hit_object.time)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def calculate_performance(
    difficulty: DifficultyAttributes,
    *,
    accuracy: float = 1.0,
    combo: int | None = None,
    misses: int = 0,
) -> PerformanceAttributes:
    accuracy = clamp(accuracy, 0.0, 1.0)
    total_hits = max(1, difficulty.hit_circle_count + difficulty.slider_count + difficulty.spinner_count)
    max_combo = max(1, difficulty.max_combo)
    misses = max(0, misses)

    if combo is None:
        combo = max_combo
    combo = int(clamp(combo, 0, max_combo))

    effective_miss_count = max(float(misses), total_hits / 200.0)

    aim_pp = difficulty_to_performance(difficulty.aim) * math.pow(accuracy, 5.5) * (0.98 + (max_combo / 1500.0))
    speed_pp = difficulty_to_performance(difficulty.speed) * math.pow(accuracy, 4.0)
    accuracy_pp = math.pow(accuracy, 5.5) * (25.0 + difficulty.star_rating * 5.0)

    if combo < max_combo:
        factor = math.pow(combo / max_combo, 0.8)
        aim_pp *= factor
        speed_pp *= factor

    miss_penalty = math.pow(0.97, effective_miss_count)
    aim_pp *= miss_penalty
    speed_pp *= miss_penalty

    total_pp = math.pow(
        math.pow(aim_pp, 1.1) + math.pow(speed_pp, 1.1) + math.pow(accuracy_pp, 1.1),
        1.0 / 1.1,
    )

    return PerformanceAttributes(
        pp=total_pp,
        aim_pp=aim_pp,
        speed_pp=speed_pp,
        accuracy_pp=accuracy_pp,
        accuracy=accuracy,
        effective_miss_count=effective_miss_count,
    )
