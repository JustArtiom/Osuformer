from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    ABS_TIME = "abs_time"
    REL_TIME = "rel_time"
    SNAPPING = "snapping"
    DISTANCE = "distance"
    POS = "pos"
    NEW_COMBO = "new_combo"
    HITSOUND = "hitsound"
    VOLUME = "volume"
    CIRCLE = "circle"
    SPINNER = "spinner"
    SPINNER_END = "spinner_end"
    SLIDER_HEAD = "slider_head"
    SLIDER_END = "slider_end"
    BEZIER_ANCHOR = "bezier_anchor"
    PERFECT_ANCHOR = "perfect_anchor"
    CATMULL_ANCHOR = "catmull_anchor"
    LINEAR_ANCHOR = "linear_anchor"
    RED_ANCHOR = "red_anchor"
    SLIDER_SLIDES = "slider_slides"
    DURATION = "duration"
    BEAT = "beat"
    MEASURE = "measure"
    TIMING_POINT = "timing_point"
    KIAI = "kiai"
    SCROLL_SPEED = "scroll_speed"
    DIFFICULTY = "difficulty"
    DESCRIPTOR = "descriptor"
    YEAR = "year"
    HITSOUNDED = "hitsounded"
    CS = "cs"
    AR = "ar"
    OD = "od"
    HP = "hp"
    GLOBAL_SV = "global_sv"
    SONG_LENGTH = "song_length"
    SONG_POSITION = "song_position"


@dataclass(frozen=True)
class Event:
    type: EventType
    value: int = 0

    def __repr__(self) -> str:
        return f"{self.type.value}({self.value})"
