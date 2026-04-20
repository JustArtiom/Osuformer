from __future__ import annotations

from dataclasses import dataclass

from src.config.schemas.tokenizer import TokenizerConfig
from src.osu.beatmap import Beatmap
from src.osu.enums import CurveType, Effects, GameMode, HitSound
from src.osu.hit_object import Circle, Slider, SliderCurve, SliderObjectParams, Spinner, SpinnerObjectParams
from src.osu.hit_sample import HitSample
from src.osu.sections import Difficulty, Events, General, Metadata
from src.osu.timing_point import TimingPoint
from src.osu_tokenizer import Event, EventType, Vocab
from src.osu_tokenizer.hitsound_codec import decode_hitsound


_ANCHOR_TO_CURVE: dict[EventType, CurveType] = {
    EventType.BEZIER_ANCHOR: CurveType.BEZIER,
    EventType.PERFECT_ANCHOR: CurveType.PERFECT,
    EventType.CATMULL_ANCHOR: CurveType.CATMULL,
    EventType.LINEAR_ANCHOR: CurveType.LINEAR,
}


@dataclass
class _TimelineState:
    beat_length: float
    sv_multiplier: float
    kiai: bool


def events_to_beatmap(
    events: list[Event],
    vocab: Vocab,
    tokenizer_cfg: TokenizerConfig,
    audio_filename: str,
    bpm: float,
    title: str = "Generated",
    artist: str = "osuformer",
    creator: str = "osuformer",
    version: str = "Generated",
    circle_size: float = 4.0,
    approach_rate: float = 9.0,
    overall_difficulty: float = 8.0,
    hp_drain_rate: float = 6.0,
    slider_multiplier: float = 1.4,
) -> Beatmap:
    groups = _group_by_abs_time(events)
    timing_points = _build_timing_points(groups, tokenizer_cfg, bpm)
    hit_objects: list = []
    i = 0
    while i < len(groups):
        abs_bin, group = groups[i]
        time_ms = abs_bin * tokenizer_cfg.dt_bin_ms
        marker = _find_marker(group)
        pos = _find_pos(group, vocab)
        hs_value = _find_value(group, EventType.HITSOUND) or 0
        volume = _find_value(group, EventType.VOLUME) or 0
        new_combo = any(ev.type == EventType.NEW_COMBO for ev in group)
        hit_sound, normal_set, addition_set = decode_hitsound(hs_value)
        hit_sample = HitSample(normal_set=normal_set, addition_set=addition_set, volume=volume)
        type_byte = 4 if new_combo else 0
        if marker == EventType.CIRCLE:
            hit_objects.append(
                Circle(
                    x=pos[0],
                    y=pos[1],
                    time=float(time_ms),
                    type=1 | type_byte,
                    hit_sound=hit_sound,
                    hit_sample=hit_sample,
                )
            )
            i += 1
        elif marker == EventType.SLIDER_HEAD:
            slider, consumed = _build_slider(
                head_time_ms=time_ms,
                head_pos=pos,
                head_hit_sound=hit_sound,
                head_sample=hit_sample,
                new_combo=new_combo,
                groups=groups,
                start_index=i,
                vocab=vocab,
                tokenizer_cfg=tokenizer_cfg,
                slider_multiplier=slider_multiplier,
                timing_points=timing_points,
            )
            if slider is not None:
                hit_objects.append(slider)
            i += consumed
        elif marker == EventType.SPINNER:
            end_time_ms = _find_spinner_end(groups, i, tokenizer_cfg)
            hit_objects.append(
                Spinner(
                    x=256.0,
                    y=192.0,
                    time=float(time_ms),
                    type=8 | type_byte,
                    hit_sound=hit_sound,
                    object_params=SpinnerObjectParams(end_time=float(end_time_ms)),
                    hit_sample=hit_sample,
                )
            )
            i += 1
        else:
            i += 1

    return Beatmap(
        general=General(audio_filename=audio_filename, mode=GameMode.STANDARD),
        metadata=Metadata(
            title=title,
            title_unicode=title,
            artist=artist,
            artist_unicode=artist,
            creator=creator,
            version=version,
        ),
        difficulty=Difficulty(
            circle_size=circle_size,
            approach_rate=approach_rate,
            overall_difficulty=overall_difficulty,
            hp_drain_rate=hp_drain_rate,
            slider_multiplier=slider_multiplier,
        ),
        events=Events(),
        timing_points=timing_points,
        hit_objects=hit_objects,
    )


def _build_timing_points(
    groups: list[tuple[int, list[Event]]],
    cfg: TokenizerConfig,
    bpm: float,
) -> list[TimingPoint]:
    base_beat_length = 60000.0 / max(1.0, bpm)
    tps: list[TimingPoint] = [
        TimingPoint(time=0.0, beat_length=base_beat_length, uninherited=1, effects=Effects.NONE)
    ]
    seen_times: set[float] = {0.0}
    for abs_bin, group in groups:
        time_ms = float(abs_bin * cfg.dt_bin_ms)
        if time_ms in seen_times:
            continue
        kiai = _find_value(group, EventType.KIAI) == 1
        effects = Effects.KIAI if kiai else Effects.NONE
        if any(ev.type == EventType.TIMING_POINT for ev in group):
            tps.append(
                TimingPoint(
                    time=time_ms,
                    beat_length=base_beat_length,
                    uninherited=1,
                    effects=effects,
                )
            )
            seen_times.add(time_ms)
            continue
        scroll_speed = _find_value(group, EventType.SCROLL_SPEED)
        if scroll_speed is not None and scroll_speed > 0:
            sv = scroll_speed / 100.0
            beat_length = -100.0 / sv
            tps.append(
                TimingPoint(
                    time=time_ms,
                    beat_length=beat_length,
                    uninherited=0,
                    effects=effects,
                )
            )
            seen_times.add(time_ms)
    tps.sort(key=lambda tp: tp.time)
    return tps


def _state_at(time_ms: float, timing_points: list[TimingPoint]) -> _TimelineState:
    beat_length = 60000.0 / 180.0
    sv = 1.0
    kiai = False
    for tp in timing_points:
        if tp.time > time_ms:
            break
        if tp.uninherited == 1 and tp.beat_length > 0:
            beat_length = tp.beat_length
            sv = 1.0
        elif tp.uninherited == 0 and tp.beat_length < 0:
            sv = -100.0 / tp.beat_length
        kiai = bool(tp.effects & Effects.KIAI)
    return _TimelineState(beat_length=beat_length, sv_multiplier=sv, kiai=kiai)


def _group_by_abs_time(events: list[Event]) -> list[tuple[int, list[Event]]]:
    groups: list[tuple[int, list[Event]]] = []
    current_abs: int | None = None
    current: list[Event] = []
    for ev in events:
        if ev.type == EventType.ABS_TIME:
            if current_abs is not None:
                groups.append((current_abs, current))
            current_abs = ev.value
            current = []
        else:
            current.append(ev)
    if current_abs is not None:
        groups.append((current_abs, current))
    return groups


def _find_marker(group: list[Event]) -> EventType | None:
    for ev in group:
        if ev.type in {
            EventType.CIRCLE,
            EventType.SLIDER_HEAD,
            EventType.SLIDER_END,
            EventType.SPINNER,
            EventType.SPINNER_END,
            EventType.LAST_ANCHOR,
        }:
            return ev.type
    return None


def _find_pos(group: list[Event], vocab: Vocab) -> tuple[float, float]:
    for ev in group:
        if ev.type == EventType.POS:
            return vocab.grid.decode(ev.value)
    return (256.0, 192.0)


def _find_value(group: list[Event], event_type: EventType) -> int | None:
    for ev in group:
        if ev.type == event_type:
            return ev.value
    return None


def _find_spinner_end(groups: list[tuple[int, list[Event]]], start_index: int, cfg: TokenizerConfig) -> float:
    head_bin = groups[start_index][0]
    for abs_bin, group in groups[start_index + 1 :]:
        for ev in group:
            if ev.type == EventType.SPINNER_END:
                return abs_bin * cfg.dt_bin_ms
    return (head_bin + 100) * cfg.dt_bin_ms


def _build_slider(
    *,
    head_time_ms: float,
    head_pos: tuple[float, float],
    head_hit_sound: HitSound,
    head_sample: HitSample,
    new_combo: bool,
    groups: list[tuple[int, list[Event]]],
    start_index: int,
    vocab: Vocab,
    tokenizer_cfg: TokenizerConfig,
    slider_multiplier: float,
    timing_points: list[TimingPoint],
) -> tuple[Slider | None, int]:
    segments: list[list[tuple[float, float]]] = []
    current_curve_type: CurveType = CurveType.LINEAR
    current_points: list[tuple[float, float]] = []
    last_anchor_pos: tuple[float, float] | None = None
    last_anchor_time_ms: float | None = None
    slides = 1
    j = start_index + 1
    while j < len(groups):
        abs_bin, group = groups[j]
        if any(ev.type == EventType.SLIDER_END for ev in group):
            for ev in group:
                if ev.type == EventType.SLIDER_SLIDES:
                    slides = max(1, min(8, ev.value))
            j += 1
            break
        if any(ev.type == EventType.LAST_ANCHOR for ev in group):
            last_anchor_pos = _find_pos(group, vocab)
            last_anchor_time_ms = float(abs_bin * tokenizer_cfg.dt_bin_ms)
            if current_points:
                current_points.append(last_anchor_pos)
                segments.append(current_points)
                current_points = []
            else:
                segments.append([last_anchor_pos])
            j += 1
            continue
        anchor_type: EventType | None = None
        pos_event: tuple[float, float] | None = None
        red_anchor = False
        for ev in group:
            if ev.type == EventType.RED_ANCHOR:
                red_anchor = True
            if ev.type in _ANCHOR_TO_CURVE:
                anchor_type = ev.type
            if ev.type == EventType.POS:
                pos_event = vocab.grid.decode(ev.value)
        if pos_event is None:
            j += 1
            continue
        if red_anchor and current_points:
            segments.append(current_points)
            current_points = []
        if anchor_type is not None:
            new_type = _ANCHOR_TO_CURVE[anchor_type]
            if new_type != current_curve_type and current_points:
                segments.append(current_points)
                current_points = []
            current_curve_type = new_type
        current_points.append(pos_event)
        j += 1

    if not segments:
        return None, j - start_index
    if last_anchor_pos is None or last_anchor_time_ms is None:
        return None, j - start_index

    curves = [SliderCurve(curve_type=_infer_curve_type(seg, current_curve_type), curve_points=seg) for seg in segments]
    intended_duration = max(1.0, last_anchor_time_ms - head_time_ms)
    state = _state_at(head_time_ms, timing_points)
    length_by_duration = (intended_duration / state.beat_length) * 100.0 * slider_multiplier * state.sv_multiplier / max(1, slides)
    length_by_geometry = _total_length(head_pos, segments)
    length = length_by_duration if length_by_duration > 1.0 else length_by_geometry
    type_byte = 2 | (4 if new_combo else 0)
    params = SliderObjectParams(curves=curves, slides=slides, length=length, duration=intended_duration)
    slider = Slider(
        x=head_pos[0],
        y=head_pos[1],
        time=float(head_time_ms),
        type=type_byte,
        hit_sound=head_hit_sound,
        object_params=params,
        hit_sample=head_sample,
    )
    return slider, j - start_index


def _infer_curve_type(points: list[tuple[float, float]], fallback: CurveType) -> CurveType:
    if len(points) == 1:
        return CurveType.LINEAR
    return fallback if fallback != CurveType.LINEAR else CurveType.BEZIER


def _total_length(head: tuple[float, float], segments: list[list[tuple[float, float]]]) -> float:
    from math import hypot

    length = 0.0
    prev = head
    for seg in segments:
        for point in seg:
            length += hypot(point[0] - prev[0], point[1] - prev[1])
            prev = point
    return length
