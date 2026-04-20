from __future__ import annotations

from src.config.loader import load_config
from src.config.schemas.tokenizer import TokenizerConfig
from src.osu.beatmap import Beatmap
from src.osu.enums import CurveType, HitSound
from src.osu.hit_object import Circle, Slider, SliderCurve, SliderObjectParams
from src.osu.sections import Difficulty, General, Metadata
from src.osu.timing_point import TimingPoint


def make_config() -> TokenizerConfig:
    return load_config("config/config.yaml").tokenizer


def make_timing_point(time: float = 0.0, bpm: float = 180.0) -> TimingPoint:
    return TimingPoint(time=time, beat_length=60000.0 / bpm, uninherited=1)


def make_circle(time: float, x: float = 256.0, y: float = 192.0, hit_sound: HitSound = HitSound.NORMAL) -> Circle:
    return Circle(time=time, x=x, y=y, type=1, hit_sound=hit_sound)


def make_slider(
    time: float,
    head: tuple[float, float] = (100.0, 100.0),
    anchors: list[tuple[float, float]] | None = None,
    duration: float = 500.0,
    slides: int = 1,
    curve_type: CurveType = CurveType.BEZIER,
) -> Slider:
    pts = anchors if anchors is not None else [(200.0, 200.0)]
    curve = SliderCurve(curve_type=curve_type, curve_points=pts)
    params = SliderObjectParams(curves=[curve], slides=slides, length=100.0, duration=duration)
    return Slider(time=time, x=head[0], y=head[1], type=2, hit_sound=HitSound.NORMAL, object_params=params)


def build_beatmap(hit_objects: list, timing_points: list[TimingPoint] | None = None) -> Beatmap:
    tps = timing_points if timing_points is not None else [make_timing_point()]
    return Beatmap(
        general=General(),
        metadata=Metadata(beatmap_id=1, beatmap_set_id=1),
        difficulty=Difficulty(circle_size=4.0),
        timing_points=tps,
        hit_objects=list(hit_objects),
    )
