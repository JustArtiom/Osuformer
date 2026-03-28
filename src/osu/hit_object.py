from __future__ import annotations
from typing import Optional, Union
import re

from .enums import CurveType, HitSound
from .hit_sample import HitSample
from .utils import fmt


class HitObject:
    def __init__(
        self,
        *,
        raw: str = "",
        x: float = 0,
        y: float = 0,
        time: float = 0,
        type: int = 0,
        hit_sound: HitSound = HitSound.NONE,
        object_params: Optional[Union["SpinnerObjectParams", "SliderObjectParams", "HoldNoteObjectParams"]] = None,
        hit_sample: Optional[HitSample] = None,
    ):
        self.x = x
        self.y = y
        self.time = time
        self.type = type
        self.hit_sound = hit_sound
        self.object_params = object_params
        self.hit_sample = hit_sample if hit_sample is not None else HitSample()

        if raw:
            self._load_raw(raw)

    def is_new_combo(self, type_override: Optional[int] = None) -> bool:
        t = type_override if type_override is not None else self.type
        return (t & 4) != 0

    def get_combo_skip_count(self, type_override: Optional[int] = None) -> int:
        t = type_override if type_override is not None else self.type
        return (t >> 4) & 0b111

    def _load_raw(self, raw: str) -> None:
        raise NotImplementedError


class Circle(HitObject):
    def __init__(
        self,
        *,
        raw: str = "",
        x: float = 0,
        y: float = 0,
        time: float = 0,
        type: int = 1,
        hit_sound: HitSound = HitSound.NONE,
        hit_sample: Optional[HitSample] = None,
    ):
        super().__init__(raw=raw, x=x, y=y, time=time, type=type, hit_sound=hit_sound, object_params=None, hit_sample=hit_sample)

    def _load_raw(self, raw: str) -> None:
        segments = [s.strip() for s in raw.split(",")]
        self.x = float(segments[0])
        self.y = float(segments[1])
        self.time = float(segments[2])
        self.type = int(segments[3])
        self.hit_sound = HitSound(int(segments[4]))
        self.hit_sample = HitSample(raw=segments[5]) if len(segments) > 5 and segments[5] else HitSample()

    def __str__(self) -> str:
        return f"{fmt(self.x)},{fmt(self.y)},{fmt(self.time)},{self.type},{int(self.hit_sound)},{self.hit_sample}"


class SliderCurve:
    def __init__(
        self,
        *,
        raw: str = "",
        curve_type: CurveType = CurveType.LINEAR,
        curve_points: Optional[list[tuple[float, float]]] = None,
        degree: Optional[int] = None,
    ):
        self.curve_type = curve_type
        self.curve_points: list[tuple[float, float]] = list(curve_points) if curve_points is not None else []
        self.degree = degree

        if raw:
            self._load_raw(raw)

    def _load_raw(self, raw: str) -> None:
        parts = raw.split("|")
        curve_type_str = parts[0]

        # Handle degree-parameterized B-splines: "B3", "B5", etc.
        if len(curve_type_str) > 1 and curve_type_str[0] == "B" and curve_type_str[1:].isdigit():
            self.curve_type = CurveType.BEZIER
            self.degree = int(curve_type_str[1:])
        else:
            self.curve_type = CurveType(curve_type_str)
            self.degree = None

        self.curve_points = [tuple(map(float, p.split(":"))) for p in parts[1:] if p]  # type: ignore[misc]

    def __str__(self) -> str:
        type_str = f"B{self.degree}" if self.degree is not None else self.curve_type.value
        if not self.curve_points:
            return type_str
        points_str = "|".join(f"{fmt(x)}:{fmt(y)}" for x, y in self.curve_points)
        return f"{type_str}|{points_str}"


class SliderObjectParams:
    def __init__(
        self,
        *,
        raw: str = "",
        curves: Optional[list[SliderCurve]] = None,
        slides: int = 1,
        length: float = 0.0,
        duration: float = 0.0,
        edge_sounds: Optional[list[HitSound]] = None,
        edge_sets: Optional[list[tuple[int, int]]] = None,
    ):
        self.curves: list[SliderCurve] = list(curves) if curves is not None else []
        self.slides = slides
        self.length = length
        self.duration = duration
        self.edge_sounds: list[HitSound] = list(edge_sounds) if edge_sounds is not None else [HitSound.NONE] * (slides + 1)
        self.edge_sets: list[tuple[int, int]] = list(edge_sets) if edge_sets is not None else [(0, 0)] * (slides + 1)

        if raw:
            self._load_raw(raw)

    def _load_raw(self, raw: str) -> None:
        segments = [s.strip() for s in raw.split(",")]

        # Split curve string into individual curve segments by finding type-letter boundaries
        pieces = segments[0].split("|")
        curve_segments: list[str] = []
        current: list[str] = []
        for piece in pieces:
            if re.match(r"^[BLCP]\d*$", piece):
                if current:
                    curve_segments.append("|".join(current))
                current = [piece]
            else:
                current.append(piece)
        if current:
            curve_segments.append("|".join(current))

        self.curves = [SliderCurve(raw=s) for s in curve_segments if s]
        self.slides = int(segments[1])
        self.length = float(segments[2])

        if len(segments) >= 4 and segments[3]:
            self.edge_sounds = [HitSound(int(v)) for v in segments[3].split("|")]
        else:
            self.edge_sounds = [HitSound.NONE] * (self.slides + 1)

        if len(segments) >= 5 and segments[4]:
            self.edge_sets = [tuple(map(int, s.split(":"))) for s in segments[4].split("|")]  # type: ignore[misc]
        else:
            self.edge_sets = [(0, 0)] * (self.slides + 1)

    def _load_duration(self, slider_velocity_multiplier: float, beat_length: float) -> None:
        self.duration = self.length * self.slides / (100.0 * slider_velocity_multiplier) * beat_length

    def __str__(self) -> str:
        curves_str = "|".join(str(c) for c in self.curves)
        edge_sounds_str = "|".join(str(int(s)) for s in self.edge_sounds)
        edge_sets_str = "|".join(f"{s[0]}:{s[1]}" for s in self.edge_sets)
        return f"{curves_str},{self.slides},{fmt(self.length)},{edge_sounds_str},{edge_sets_str}"


class Slider(HitObject):
    object_params: SliderObjectParams

    def __init__(
        self,
        *,
        raw: str = "",
        x: float = 0,
        y: float = 0,
        time: float = 0,
        type: int = 2,
        hit_sound: HitSound = HitSound.NONE,
        object_params: Optional[SliderObjectParams] = None,
        hit_sample: Optional[HitSample] = None,
    ):
        if object_params is None:
            object_params = SliderObjectParams()
        super().__init__(raw=raw, x=x, y=y, time=time, type=type, hit_sound=hit_sound, object_params=object_params, hit_sample=hit_sample)

    def _load_raw(self, raw: str) -> None:
        segments = [s.strip() for s in raw.split(",")]
        x, y, time, type_, hit_sound, *object_params_parts, last = segments

        # If the last segment has no ":", it's part of object params (no hit_sample present)
        if ":" not in last:
            object_params_parts.append(last)
            last = "0:0:0:0:"

        self.x = float(x)
        self.y = float(y)
        self.time = float(time)
        self.type = int(type_)
        self.hit_sound = HitSound(int(hit_sound))
        self.object_params = SliderObjectParams(raw=",".join(object_params_parts))
        self.hit_sample = HitSample(raw=last)

    def __str__(self) -> str:
        return f"{fmt(self.x)},{fmt(self.y)},{fmt(self.time)},{self.type},{int(self.hit_sound)},{self.object_params},{self.hit_sample}"


class SpinnerObjectParams:
    def __init__(self, *, raw: str = "", end_time: float = 0):
        self.end_time = end_time
        if raw:
            self._load_raw(raw)

    def _load_raw(self, raw: str) -> None:
        self.end_time = float(raw.strip())

    def __str__(self) -> str:
        return fmt(self.end_time)


class Spinner(HitObject):
    object_params: SpinnerObjectParams

    def __init__(
        self,
        *,
        raw: str = "",
        x: float = 256,
        y: float = 192,
        time: float = 0,
        type: int = 8,
        hit_sound: HitSound = HitSound.NONE,
        object_params: Optional[SpinnerObjectParams] = None,
        hit_sample: Optional[HitSample] = None,
    ):
        if object_params is None:
            object_params = SpinnerObjectParams(end_time=time)
        super().__init__(raw=raw, x=x, y=y, time=time, type=type, hit_sound=hit_sound, object_params=object_params, hit_sample=hit_sample)

    def _load_raw(self, raw: str) -> None:
        segments = [s.strip() for s in raw.split(",")]
        x, y, time, type_, hit_sound, *object_params_parts, last = segments

        if ":" not in last:
            object_params_parts.append(last)
            last = "0:0:0:0:"

        self.x = float(x)
        self.y = float(y)
        self.time = float(time)
        self.type = int(type_)
        self.hit_sound = HitSound(int(hit_sound))
        self.object_params = SpinnerObjectParams(raw=",".join(object_params_parts))
        self.hit_sample = HitSample(raw=last)

    def __str__(self) -> str:
        return f"{fmt(self.x)},{fmt(self.y)},{fmt(self.time)},{self.type},{int(self.hit_sound)},{self.object_params},{self.hit_sample}"


class HoldNoteObjectParams:
    def __init__(self, *, raw: str = "", end_time: float = 0):
        self.end_time = end_time
        if raw:
            self._load_raw(raw)

    def _load_raw(self, raw: str) -> None:
        self.end_time = float(raw.strip())

    def __str__(self) -> str:
        return fmt(self.end_time)


class HoldNote(HitObject):
    object_params: HoldNoteObjectParams

    def __init__(
        self,
        *,
        raw: str = "",
        x: float = 0,
        y: float = 192,
        time: float = 0,
        type: int = 128,
        hit_sound: HitSound = HitSound.NONE,
        object_params: Optional[HoldNoteObjectParams] = None,
        hit_sample: Optional[HitSample] = None,
    ):
        if object_params is None:
            object_params = HoldNoteObjectParams(end_time=time)
        super().__init__(raw=raw, x=x, y=y, time=time, type=type, hit_sound=hit_sound, object_params=object_params, hit_sample=hit_sample)

    def _load_raw(self, raw: str) -> None:
        segments = [s.strip() for s in raw.split(",")]
        self.x = float(segments[0])
        self.y = float(segments[1])
        self.time = float(segments[2])
        self.type = int(segments[3])
        self.hit_sound = HitSound(int(segments[4]))

        # Format: endTime:normalSet:additionSet:sampleIndex:sampleVolume:filename
        last_part = segments[5] if len(segments) > 5 else "0:0:0:0:0:"
        if ":" in last_part:
            colon_idx = last_part.index(":")
            self.object_params = HoldNoteObjectParams(end_time=float(last_part[:colon_idx]))
            self.hit_sample = HitSample(raw=last_part[colon_idx + 1:])
        else:
            self.object_params = HoldNoteObjectParams(end_time=float(last_part))
            self.hit_sample = HitSample()

    def __str__(self) -> str:
        return f"{fmt(self.x)},{fmt(self.y)},{fmt(self.time)},{self.type},{int(self.hit_sound)},{fmt(self.object_params.end_time)}:{self.hit_sample}"
