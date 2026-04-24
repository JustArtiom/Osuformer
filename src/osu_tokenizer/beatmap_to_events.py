from __future__ import annotations

import math
from dataclasses import dataclass

from src.config.schemas.tokenizer import TokenizerConfig
from src.osu.beatmap import Beatmap
from src.osu.enums import CurveType
from src.osu.hit_object import Circle, Slider, Spinner

from .durations import beats_to_duration_index
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
    snap_max = config.snap_max

    prev_x: float | None = None
    prev_y: float | None = None

    for obj in beatmap.hit_objects:
        obj_time_ms = float(obj.time) - window_start_ms
        abs_bin = int(round(obj_time_ms / dt_bin))
        if clamp_abs_time:
            abs_bin = max(0, min(max_time_bin, abs_bin))

        snap = _compute_snap(float(obj.time), beatmap, snap_max)

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
                    snap=snap,
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
                    snap=snap,
                    dist_px=dist_px,
                    pos_value=pos_value,
                    hs_value=hs_value,
                    volume=volume,
                    new_combo=obj.is_new_combo(),
                    vocab=vocab,
                    config=config,
                    window_start_ms=window_start_ms,
                    clamp_abs_time=clamp_abs_time,
                    beatmap=beatmap,
                )
            )
            end_x, end_y = _slider_end_position(obj)
            prev_x, prev_y = end_x, end_y
        elif isinstance(obj, Spinner):
            events.extend(
                _spinner_group(
                    spinner=obj,
                    abs_bin=abs_bin,
                    snap=snap,
                    dist_px=dist_px,
                    pos_value=pos_value,
                    hs_value=hs_value,
                    volume=volume,
                    new_combo=obj.is_new_combo(),
                    vocab=vocab,
                    config=config,
                    window_start_ms=window_start_ms,
                    clamp_abs_time=clamp_abs_time,
                    beatmap=beatmap,
                )
            )
            prev_x, prev_y = 256.0, 192.0

    return EventStream(events=events)


def _compute_snap(time_ms: float, beatmap: Beatmap, snap_max: int) -> int:
    tp = _active_uninherited(time_ms, beatmap)
    if tp is None or tp.beat_length <= 0:
        return 0
    beats = (time_ms - float(tp.time)) / tp.beat_length
    for i in range(1, snap_max + 1):
        if abs(beats - round(beats * i) / i) * tp.beat_length < 2.0:
            return i
    return 0


def _active_uninherited(time_ms: float, beatmap: Beatmap):
    active = None
    for tp in beatmap.timing_points:
        if tp.uninherited != 1 or tp.beat_length <= 0:
            continue
        if float(tp.time) <= time_ms + 1e-6:
            active = tp
        else:
            break
    if active is None:
        for tp in beatmap.timing_points:
            if tp.uninherited == 1 and tp.beat_length > 0:
                return tp
    return active


def _object_group(
    *,
    marker: EventType,
    abs_bin: int,
    snap: int,
    dist_px: int,
    pos_value: int,
    hs_value: int,
    volume: int,
    new_combo: bool,
) -> list[Event]:
    out: list[Event] = [
        Event(type=marker, value=0),
        Event(type=EventType.ABS_TIME, value=abs_bin),
        Event(type=EventType.SNAPPING, value=snap),
        Event(type=EventType.DISTANCE, value=dist_px),
        Event(type=EventType.POS, value=pos_value),
        Event(type=EventType.HITSOUND, value=hs_value),
        Event(type=EventType.VOLUME, value=volume),
    ]
    if new_combo:
        out.append(Event(type=EventType.NEW_COMBO, value=0))
    return out


def _spinner_group(
    *,
    spinner: Spinner,
    abs_bin: int,
    snap: int,
    dist_px: int,
    pos_value: int,
    hs_value: int,
    volume: int,
    new_combo: bool,
    vocab: Vocab,
    config: TokenizerConfig,
    window_start_ms: float,
    clamp_abs_time: bool = True,
    beatmap: Beatmap,
) -> list[Event]:
    del vocab, clamp_abs_time, window_start_ms
    head = _object_group(
        marker=EventType.SPINNER,
        abs_bin=abs_bin,
        snap=snap,
        dist_px=dist_px,
        pos_value=pos_value,
        hs_value=hs_value,
        volume=volume,
        new_combo=new_combo,
    )
    duration_ms = float(spinner.object_params.end_time) - float(spinner.time)
    duration_idx = _duration_index_at(float(spinner.time), duration_ms, beatmap, config)
    return head + [
        Event(type=EventType.DURATION, value=duration_idx),
        Event(type=EventType.SPINNER_END, value=0),
    ]


def _slider_group(
    *,
    slider: Slider,
    abs_bin: int,
    snap: int,
    dist_px: int,
    pos_value: int,
    hs_value: int,
    volume: int,
    new_combo: bool,
    vocab: Vocab,
    config: TokenizerConfig,
    window_start_ms: float,
    clamp_abs_time: bool = True,
    beatmap: Beatmap,
) -> list[Event]:
    del clamp_abs_time, window_start_ms
    head = _object_group(
        marker=EventType.SLIDER_HEAD,
        abs_bin=abs_bin,
        snap=snap,
        dist_px=dist_px,
        pos_value=pos_value,
        hs_value=hs_value,
        volume=volume,
        new_combo=new_combo,
    )

    grid = vocab.grid
    anchor_events: list[Event] = []
    for segment_idx, curve in enumerate(slider.object_params.curves):
        anchor_type = _anchor_type_for_curve(curve.curve_type)
        points = curve.curve_points
        for i, point in enumerate(points):
            xy = (float(point[0]), float(point[1]))
            if i == 0 and segment_idx > 0:
                anchor_events.append(Event(type=EventType.RED_ANCHOR, value=0))
                anchor_events.append(Event(type=EventType.POS, value=grid.encode(*xy)))
                continue
            anchor_events.append(Event(type=anchor_type, value=0))
            anchor_events.append(Event(type=EventType.POS, value=grid.encode(*xy)))

    duration_ms = float(slider.object_params.duration)
    duration_idx = _duration_index_at(float(slider.time), duration_ms, beatmap, config)

    slides_events: list[Event] = []
    slides = max(1, min(config.slider_slides_max, slider.object_params.slides))
    if slides > 1:
        slides_events.append(Event(type=EventType.SLIDER_SLIDES, value=slides))

    tail: list[Event] = [Event(type=EventType.DURATION, value=duration_idx)]
    tail.extend(slides_events)
    tail.append(Event(type=EventType.SLIDER_END, value=0))
    return head + anchor_events + tail


def _duration_index_at(start_time_ms: float, duration_ms: float, beatmap: Beatmap, config: TokenizerConfig) -> int:
    del config
    tp = _active_uninherited(start_time_ms, beatmap)
    if tp is None or tp.beat_length <= 0 or duration_ms <= 0:
        return beats_to_duration_index(1.0)
    beats = duration_ms / tp.beat_length
    return beats_to_duration_index(beats)


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

    def clamp_bin(t_ms: float) -> int | None:
        bin_val = int(round(t_ms / dt_bin))
        if clamp_abs_time:
            if t_ms < 0 or bin_val > max_bin:
                return None
            return max(0, min(max_bin, bin_val))
        return bin_val

    last_time_ms = _last_event_time_ms(beatmap)
    uninherited = [tp for tp in beatmap.timing_points if tp.uninherited == 1 and tp.beat_length > 0]
    for tp in beatmap.timing_points:
        t_local = float(tp.time) - window_start_ms
        bin_val = clamp_bin(t_local)
        if bin_val is None:
            continue
        out.append(Event(type=EventType.ABS_TIME, value=bin_val))
        if tp.uninherited == 1:
            out.append(Event(type=EventType.TIMING_POINT, value=0))
        else:
            sv_value = int(round(-100.0 / tp.beat_length * 100))
            sv_value = max(0, min(config.scroll_speed_max, sv_value))
            out.append(Event(type=EventType.SCROLL_SPEED, value=sv_value))
        out.append(Event(type=EventType.KIAI, value=1 if tp.is_kiai else 0))

    for idx, tp in enumerate(uninherited):
        section_start_ms = float(tp.time)
        section_end_ms = (
            float(uninherited[idx + 1].time) if idx + 1 < len(uninherited) else last_time_ms + tp.beat_length
        )
        meter = max(1, int(tp.meter))
        beat_index = 1
        time_ms = section_start_ms + tp.beat_length
        while time_ms < section_end_ms - 0.5:
            t_local = time_ms - window_start_ms
            bin_val = clamp_bin(t_local)
            if bin_val is not None:
                out.append(Event(type=EventType.ABS_TIME, value=bin_val))
                if beat_index % meter == 0:
                    out.append(Event(type=EventType.MEASURE, value=0))
                else:
                    out.append(Event(type=EventType.BEAT, value=0))
            beat_index += 1
            time_ms = section_start_ms + beat_index * tp.beat_length

    return out


def _last_event_time_ms(beatmap: Beatmap) -> float:
    last_ms = 0.0
    for obj in beatmap.hit_objects:
        last_ms = max(last_ms, float(obj.time))
        end = getattr(getattr(obj, "object_params", None), "end_time", None)
        if end is not None:
            last_ms = max(last_ms, float(end))
        duration = getattr(getattr(obj, "object_params", None), "duration", None)
        if duration is not None:
            last_ms = max(last_ms, float(obj.time) + float(duration))
    for tp in beatmap.timing_points:
        last_ms = max(last_ms, float(tp.time))
    return last_ms


_OBJECT_MARKERS: frozenset[EventType] = frozenset(
    {EventType.CIRCLE, EventType.SLIDER_HEAD, EventType.SPINNER}
)


def merge_by_time(*streams: list[Event]) -> list[Event]:
    groups: list[tuple[int, int, list[Event]]] = []
    order = 0
    for stream in streams:
        current: list[Event] = []
        current_abs: int | None = None
        pending_marker: Event | None = None
        for ev in stream:
            if ev.type in _OBJECT_MARKERS:
                if current:
                    groups.append((current_abs if current_abs is not None else 0, order, current))
                    order += 1
                current = [ev]
                current_abs = None
                pending_marker = ev
                continue
            if ev.type == EventType.ABS_TIME:
                if pending_marker is not None:
                    current.append(ev)
                    current_abs = ev.value
                    pending_marker = None
                    continue
                if current:
                    groups.append((current_abs if current_abs is not None else 0, order, current))
                    order += 1
                current = [ev]
                current_abs = ev.value
                continue
            current.append(ev)
        if current:
            groups.append((current_abs if current_abs is not None else 0, order, current))
            order += 1
    groups.sort(key=lambda g: (g[0], g[1]))
    merged: list[Event] = []
    for _, _, g in groups:
        merged.extend(g)
    return merged


__all__ = [
    "EventStream",
    "beatmap_to_events",
    "attach_rel_times",
    "collect_timing_events",
    "merge_by_time",
]
