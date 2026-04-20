from __future__ import annotations

import math
from dataclasses import dataclass

from src.config.schemas.tokenizer import TokenizerConfig
from src.osu.beatmap import Beatmap
from src.osu.enums import CurveType
from src.osu.hit_object import Circle, Slider, Spinner

from .events import Event, EventType
from .hitsound_codec import encode_hitsound
from .vocab import Vocab


@dataclass
class EventStream:
    events: list[Event]


def beatmap_to_events(
    beatmap: Beatmap,
    window_start_ms: float,
    vocab: Vocab,
    config: TokenizerConfig,
    clamp_abs_time: bool = True,
) -> EventStream:
    events: list[Event] = []
    grid = vocab.grid
    dt_bin = config.dt_bin_ms
    max_time_bin = vocab.range_for(EventType.ABS_TIME).max_value

    prev_x: float | None = None
    prev_y: float | None = None

    for obj in beatmap.hit_objects:
        obj_time_ms = float(obj.time) - window_start_ms
        abs_bin = int(round(obj_time_ms / dt_bin))
        if clamp_abs_time:
            abs_bin = max(0, min(max_time_bin, abs_bin))

        x, y = float(obj.x), float(obj.y)
        dist_px = 0
        if prev_x is not None and prev_y is not None:
            dist_px = int(round(math.hypot(x - prev_x, y - prev_y)))
        dist_px = max(0, min(config.distance_max_px, dist_px))

        pos_value = grid.encode(x, y)

        hs_sample = obj.hit_sample
        hs_value = encode_hitsound(obj.hit_sound, hs_sample.normal_set, hs_sample.addition_set)
        volume = max(0, min(config.volume_max, hs_sample.volume or config.volume_max))

        if isinstance(obj, Circle):
            events.extend(
                _object_group(
                    marker=EventType.CIRCLE,
                    abs_bin=abs_bin,
                    dist_px=dist_px,
                    pos_value=pos_value,
                    hs_value=hs_value,
                    volume=volume,
                    new_combo=obj.is_new_combo(),
                )
            )
            prev_x, prev_y = x, y
        elif isinstance(obj, Slider):
            events.extend(
                _slider_group(
                    slider=obj,
                    abs_bin=abs_bin,
                    dist_px=dist_px,
                    pos_value=pos_value,
                    hs_value=hs_value,
                    volume=volume,
                    new_combo=obj.is_new_combo(),
                    vocab=vocab,
                    config=config,
                    window_start_ms=window_start_ms,
                    clamp_abs_time=clamp_abs_time,
                )
            )
            end_x, end_y = _slider_end_position(obj)
            prev_x, prev_y = end_x, end_y
        elif isinstance(obj, Spinner):
            events.extend(
                _spinner_group(
                    spinner=obj,
                    abs_bin=abs_bin,
                    dist_px=dist_px,
                    pos_value=pos_value,
                    hs_value=hs_value,
                    volume=volume,
                    new_combo=obj.is_new_combo(),
                    vocab=vocab,
                    config=config,
                    window_start_ms=window_start_ms,
                    clamp_abs_time=clamp_abs_time,
                )
            )
            prev_x, prev_y = 256.0, 192.0

    return EventStream(events=events)


def _object_group(
    *,
    marker: EventType,
    abs_bin: int,
    dist_px: int,
    pos_value: int,
    hs_value: int,
    volume: int,
    new_combo: bool,
) -> list[Event]:
    out: list[Event] = [
        Event(type=EventType.ABS_TIME, value=abs_bin),
        Event(type=EventType.DISTANCE, value=dist_px),
        Event(type=EventType.POS, value=pos_value),
        Event(type=EventType.HITSOUND, value=hs_value),
        Event(type=EventType.VOLUME, value=volume),
    ]
    if new_combo:
        out.append(Event(type=EventType.NEW_COMBO, value=0))
    out.append(Event(type=marker, value=0))
    return out


def _spinner_group(
    *,
    spinner: Spinner,
    abs_bin: int,
    dist_px: int,
    pos_value: int,
    hs_value: int,
    volume: int,
    new_combo: bool,
    vocab: Vocab,
    config: TokenizerConfig,
    window_start_ms: float,
    clamp_abs_time: bool = True,
) -> list[Event]:
    head = _object_group(
        marker=EventType.SPINNER,
        abs_bin=abs_bin,
        dist_px=dist_px,
        pos_value=pos_value,
        hs_value=hs_value,
        volume=volume,
        new_combo=new_combo,
    )
    end_time_ms = float(spinner.object_params.end_time) - window_start_ms
    end_bin = int(round(end_time_ms / config.dt_bin_ms))
    if clamp_abs_time:
        max_bin = vocab.range_for(EventType.ABS_TIME).max_value
        end_bin = max(0, min(max_bin, end_bin))
    return head + [
        Event(type=EventType.ABS_TIME, value=end_bin),
        Event(type=EventType.SPINNER_END, value=0),
    ]


def _slider_group(
    *,
    slider: Slider,
    abs_bin: int,
    dist_px: int,
    pos_value: int,
    hs_value: int,
    volume: int,
    new_combo: bool,
    vocab: Vocab,
    config: TokenizerConfig,
    window_start_ms: float,
    clamp_abs_time: bool = True,
) -> list[Event]:
    head = _object_group(
        marker=EventType.SLIDER_HEAD,
        abs_bin=abs_bin,
        dist_px=dist_px,
        pos_value=pos_value,
        hs_value=hs_value,
        volume=volume,
        new_combo=new_combo,
    )

    grid = vocab.grid
    anchor_events: list[Event] = []
    head_xy = (float(slider.x), float(slider.y))
    last_xy = head_xy
    for segment_idx, curve in enumerate(slider.object_params.curves):
        anchor_type = _anchor_type_for_curve(curve.curve_type)
        points = curve.curve_points
        for i, point in enumerate(points):
            is_last_in_segment = i == len(points) - 1
            xy = (float(point[0]), float(point[1]))
            if i == 0 and segment_idx > 0:
                anchor_events.append(Event(type=EventType.RED_ANCHOR, value=0))
                anchor_events.append(Event(type=EventType.POS, value=grid.encode(*xy)))
                last_xy = xy
                continue
            if segment_idx == len(slider.object_params.curves) - 1 and is_last_in_segment:
                continue
            anchor_events.append(Event(type=anchor_type, value=0))
            anchor_events.append(Event(type=EventType.POS, value=grid.encode(*xy)))
            last_xy = xy

    last_curve = slider.object_params.curves[-1] if slider.object_params.curves else None
    if last_curve is not None and last_curve.curve_points:
        last_point = last_curve.curve_points[-1]
        last_xy = (float(last_point[0]), float(last_point[1]))

    end_time_ms = float(slider.time) + float(slider.object_params.duration) - window_start_ms
    end_bin = int(round(end_time_ms / config.dt_bin_ms))
    if clamp_abs_time:
        max_bin = vocab.range_for(EventType.ABS_TIME).max_value
        end_bin = max(0, min(max_bin, end_bin))
    last_dist = int(round(math.hypot(last_xy[0] - head_xy[0], last_xy[1] - head_xy[1])))
    last_dist = max(0, min(config.distance_max_px, last_dist))

    last_anchor = [
        Event(type=EventType.ABS_TIME, value=end_bin),
        Event(type=EventType.DISTANCE, value=last_dist),
        Event(type=EventType.POS, value=grid.encode(*last_xy)),
        Event(type=EventType.LAST_ANCHOR, value=0),
    ]

    slides_events: list[Event] = []
    slides = max(1, min(config.slider_slides_max, slider.object_params.slides))
    if slides > 1:
        slides_events.append(Event(type=EventType.SLIDER_SLIDES, value=slides))

    return head + anchor_events + last_anchor + slides_events + [Event(type=EventType.SLIDER_END, value=0)]


def _anchor_type_for_curve(curve_type: CurveType) -> EventType:
    if curve_type == CurveType.BEZIER:
        return EventType.BEZIER_ANCHOR
    if curve_type == CurveType.PERFECT:
        return EventType.PERFECT_ANCHOR
    if curve_type == CurveType.CATMULL:
        return EventType.CATMULL_ANCHOR
    return EventType.LINEAR_ANCHOR


def _slider_end_position(slider: Slider) -> tuple[float, float]:
    curves = slider.object_params.curves
    if not curves or not curves[-1].curve_points:
        return float(slider.x), float(slider.y)
    last = curves[-1].curve_points[-1]
    return float(last[0]), float(last[1])


def attach_rel_times(events: list[Event], vocab: Vocab, config: TokenizerConfig) -> list[Event]:
    out: list[Event] = []
    last_abs_bin: int | None = None
    max_bin = vocab.range_for(EventType.REL_TIME).max_value
    for event in events:
        out.append(event)
        if event.type == EventType.ABS_TIME:
            if last_abs_bin is None:
                rel = 0
            else:
                delta = event.value - last_abs_bin
                if delta < 0 or delta > max_bin:
                    rel = max_bin
                else:
                    rel = delta
            out.append(Event(type=EventType.REL_TIME, value=rel))
            last_abs_bin = event.value
    return out


def collect_timing_events(
    beatmap: Beatmap,
    window_start_ms: float,
    vocab: Vocab,
    config: TokenizerConfig,
    clamp_abs_time: bool = True,
) -> list[Event]:
    out: list[Event] = []
    dt_bin = config.dt_bin_ms
    max_bin = vocab.range_for(EventType.ABS_TIME).max_value
    for tp in beatmap.timing_points:
        t = float(tp.time) - window_start_ms
        abs_bin = int(round(t / dt_bin))
        if clamp_abs_time:
            if t < 0 or abs_bin > max_bin:
                continue
            abs_bin = max(0, min(max_bin, abs_bin))
        out.append(Event(type=EventType.ABS_TIME, value=abs_bin))
        if tp.uninherited == 1:
            out.append(Event(type=EventType.TIMING_POINT, value=0))
        else:
            sv_value = int(round(-100.0 / tp.beat_length * 100))
            sv_value = max(0, min(config.scroll_speed_max, sv_value))
            out.append(Event(type=EventType.SCROLL_SPEED, value=sv_value))
        out.append(Event(type=EventType.KIAI, value=1 if tp.is_kiai else 0))
    return out


def merge_by_time(*streams: list[Event]) -> list[Event]:
    groups: list[tuple[int, list[Event]]] = []
    for stream in streams:
        current: list[Event] = []
        current_abs: int | None = None
        for ev in stream:
            if ev.type == EventType.ABS_TIME:
                if current:
                    groups.append((current_abs if current_abs is not None else 0, current))
                current = [ev]
                current_abs = ev.value
            else:
                current.append(ev)
        if current:
            groups.append((current_abs if current_abs is not None else 0, current))
    groups.sort(key=lambda g: g[0])
    merged: list[Event] = []
    for _, g in groups:
        merged.extend(g)
    return merged


__all__ = [
    "EventStream",
    "beatmap_to_events",
    "attach_rel_times",
    "collect_timing_events",
    "merge_by_time",
]
