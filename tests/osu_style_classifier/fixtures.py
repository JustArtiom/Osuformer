from __future__ import annotations

from typing import Sequence

from src.osu.beatmap import Beatmap, HitObjectType
from src.osu.hit_object import Circle, Slider, SliderCurve, SliderObjectParams, Spinner, SpinnerObjectParams
from src.osu.enums import CurveType, HitSound
from src.osu.sections import General, Difficulty, Metadata
from src.osu.timing_point import TimingPoint


def make_timing_point(time: float = 0, bpm: float = 180.0, meter: int = 4) -> TimingPoint:
    return TimingPoint(time=time, beat_length=60000.0 / bpm, meter=meter, uninherited=1)


def make_inherited_tp(time: float, sv_multiplier: float) -> TimingPoint:
    return TimingPoint(time=time, beat_length=-100.0 / sv_multiplier, uninherited=0)


def make_circle(time: float, x: float = 256, y: float = 192) -> Circle:
    return Circle(time=time, x=x, y=y, type=1, hit_sound=HitSound.NORMAL)


def make_slider(
    time: float,
    x: float = 0,
    y: float = 0,
    length: float = 100.0,
    slides: int = 1,
    anchors: int = 1,
    duration: float = 250.0,
) -> Slider:
    curve_points = [(x + 20.0 * (i + 1), y) for i in range(anchors)]
    curve = SliderCurve(curve_type=CurveType.LINEAR, curve_points=curve_points)
    params = SliderObjectParams(curves=[curve], slides=slides, length=length, duration=duration)
    return Slider(time=time, x=x, y=y, type=2, hit_sound=HitSound.NORMAL, object_params=params)


def make_spinner(time: float, end_time: float) -> Spinner:
    return Spinner(time=time, object_params=SpinnerObjectParams(end_time=end_time))


def build_beatmap(
    hit_objects: Sequence[HitObjectType],
    timing_points: list[TimingPoint] | None = None,
    circle_size: float = 4.0,
    approach_rate: float = 9.0,
) -> Beatmap:
    tps = timing_points if timing_points is not None else [make_timing_point()]
    return Beatmap(
        general=General(),
        metadata=Metadata(beatmap_id=1, beatmap_set_id=1),
        difficulty=Difficulty(circle_size=circle_size, approach_rate=approach_rate),
        timing_points=tps,
        hit_objects=list(hit_objects),
    )


def stream_circles(count: int, start_ms: float, interval_ms: float, x0: float = 100.0, dx: float = 20.0) -> list[Circle]:
    return [make_circle(start_ms + i * interval_ms, x=x0 + i * dx, y=192) for i in range(count)]


def jump_circles(count: int, start_ms: float, interval_ms: float) -> list[Circle]:
    out: list[Circle] = []
    positions = [(80, 80), (420, 300), (80, 300), (420, 80)]
    for i in range(count):
        x, y = positions[i % 4]
        out.append(make_circle(start_ms + i * interval_ms, x=float(x), y=float(y)))
    return out
