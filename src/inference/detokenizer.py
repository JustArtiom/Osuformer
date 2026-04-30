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
from src.osu_tokenizer.durations import duration_index_to_beats
from src.osu_tokenizer.hitsound_codec import decode_hitsound

_OBJECT_MARKERS: frozenset[EventType] = frozenset(
    {EventType.CIRCLE, EventType.SLIDER_HEAD, EventType.SPINNER}
)


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
    auto_timing: bool = False,
    offset_ms: float | None = None,
) -> Beatmap:
    groups = _group_by_abs_time(events)
    if auto_timing:
        timing_points = _build_timing_points_from_beats(groups, tokenizer_cfg, bpm, offset_ms)
    else:
        timing_points = _build_timing_points(groups, tokenizer_cfg, bpm, offset_ms)
    hit_objects: list = []
    i = 0
    while i < len(groups):
        abs_bin, group = groups[i]
        raw_time_ms = float(abs_bin * tokenizer_cfg.dt_bin_ms)
        snap = _find_value(group, EventType.SNAPPING) or 0
        time_ms = _resnap_time_to_divisor(raw_time_ms, snap, timing_points)
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
            slider = _build_slider(
                head_time_ms=time_ms,
                head_pos=pos,
                head_hit_sound=hit_sound,
                head_sample=hit_sample,
                new_combo=new_combo,
                group=group,
                vocab=vocab,
                slider_multiplier=slider_multiplier,
                timing_points=timing_points,
            )
            if slider is not None:
                hit_objects.append(slider)
            i += 1
        elif marker == EventType.SPINNER:
            end_time_ms = _find_spinner_end(group, time_ms, timing_points)
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
    offset_ms: float | None = None,
) -> list[TimingPoint]:
    inferred_beat_length = _infer_beat_length_ms(groups, cfg)
    if bpm <= 0 and inferred_beat_length is None:
        raise ValueError("No BPM provided and no BEAT tokens emitted to infer from.")
    base_beat_length = inferred_beat_length if (bpm <= 0 and inferred_beat_length is not None) else 60000.0 / max(1.0, bpm)
    primary_offset = float(offset_ms) if offset_ms is not None else 0.0
    user_override = offset_ms is not None and bpm > 0

    section_starts = [primary_offset] + [
        float(abs_bin * cfg.dt_bin_ms)
        for abs_bin, group in groups
        if any(ev.type == EventType.TIMING_POINT for ev in group)
    ]
    section_starts = sorted(set(section_starts))
    if user_override:
        section_bpms: dict[float, float] = {s: base_beat_length for s in section_starts}
    else:
        section_bpms = _per_section_beat_lengths(groups, cfg, section_starts, base_beat_length)

    tps: list[TimingPoint] = [
        TimingPoint(
            time=primary_offset,
            beat_length=section_bpms.get(primary_offset, base_beat_length),
            uninherited=1,
            effects=Effects.NONE,
        )
    ]
    seen_times: set[float] = {primary_offset}
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
                    beat_length=section_bpms.get(time_ms, base_beat_length),
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


def _per_section_beat_lengths(
    groups: list[tuple[int, list[Event]]],
    cfg: TokenizerConfig,
    section_starts: list[float],
    fallback_beat_length: float,
) -> dict[float, float]:
    """For each TIMING_POINT section start, derive BPM from BEAT events in that section."""
    beat_times: list[float] = []
    for abs_bin, group in groups:
        if any(ev.type in (EventType.BEAT, EventType.MEASURE) for ev in group):
            beat_times.append(float(abs_bin * cfg.dt_bin_ms))
    beat_times.sort()
    if not section_starts:
        return {}
    out: dict[float, float] = {}
    for i, start in enumerate(section_starts):
        end = section_starts[i + 1] if i + 1 < len(section_starts) else float("inf")
        section_beats = [t for t in beat_times if start <= t < end]
        if len(section_beats) < 4:
            out[start] = fallback_beat_length
            continue
        deltas = sorted(section_beats[j + 1] - section_beats[j] for j in range(len(section_beats) - 1))
        median_delta = deltas[len(deltas) // 2]
        if not 100.0 < median_delta < 2000.0:
            out[start] = fallback_beat_length
            continue
        refined = _refine_beat_length(section_beats, median_delta)
        out[start] = refined if refined is not None else median_delta
    return out


def _build_timing_points_from_beats(
    groups: list[tuple[int, list[Event]]],
    cfg: TokenizerConfig,
    bpm: float,
    offset_ms: float | None = None,
) -> list[TimingPoint]:
    beat_events: list[tuple[float, bool]] = []
    for abs_bin, group in groups:
        time_ms = float(abs_bin * cfg.dt_bin_ms)
        is_measure = any(ev.type == EventType.MEASURE for ev in group)
        is_beat = any(ev.type == EventType.BEAT for ev in group) or is_measure
        if is_beat:
            beat_events.append((time_ms, is_measure))
    if offset_ms is not None and bpm > 0:
        beat_length = 60000.0 / max(1.0, bpm)
        tps: list[TimingPoint] = [
            TimingPoint(time=float(offset_ms), beat_length=beat_length, uninherited=1, effects=Effects.NONE)
        ]
        seen_times: set[float] = {float(offset_ms)}
        _append_inherited_points(tps, groups, cfg, seen_times)
        tps.sort(key=lambda tp: tp.time)
        return tps
    if len(beat_events) < 4:
        if bpm <= 0:
            raise ValueError("auto-timing needs at least 4 BEAT/MEASURE events or an explicit --bpm")
        primary_offset = float(offset_ms) if offset_ms is not None else 0.0
        return [TimingPoint(time=primary_offset, beat_length=60000.0 / max(1.0, bpm), uninherited=1, effects=Effects.NONE)]
    beat_events.sort(key=lambda x: x[0])
    times = [t for t, _ in beat_events]
    deltas = sorted(times[i + 1] - times[i] for i in range(len(times) - 1))
    median_delta = deltas[len(deltas) // 2]
    if not 100.0 < median_delta < 2000.0:
        median_delta = 60000.0 / max(1.0, bpm) if bpm > 0 else 333.33
    refined = _refine_beat_length(times, median_delta)
    if refined is not None:
        median_delta = refined
    if offset_ms is not None:
        offset = float(offset_ms)
    else:
        measure_times = [t for t, m in beat_events if m]
        anchor_time = measure_times[0] if measure_times else times[0]
        offset = anchor_time
        while offset - median_delta * 4 >= 0:
            offset -= median_delta * 4
    tps: list[TimingPoint] = [
        TimingPoint(time=offset, beat_length=median_delta, uninherited=1, effects=Effects.NONE)
    ]
    seen_times: set[float] = {offset}
    _append_inherited_points(tps, groups, cfg, seen_times)
    tps.sort(key=lambda tp: tp.time)
    return tps


def _append_inherited_points(
    tps: list[TimingPoint],
    groups: list[tuple[int, list[Event]]],
    cfg: TokenizerConfig,
    seen_times: set[float],
) -> None:
    for abs_bin, group in groups:
        time_ms = float(abs_bin * cfg.dt_bin_ms)
        if time_ms in seen_times:
            continue
        kiai = _find_value(group, EventType.KIAI) == 1
        effects = Effects.KIAI if kiai else Effects.NONE
        scroll_speed = _find_value(group, EventType.SCROLL_SPEED)
        if scroll_speed is not None and scroll_speed > 0:
            sv = scroll_speed / 100.0
            beat_length = -100.0 / sv
            tps.append(
                TimingPoint(time=time_ms, beat_length=beat_length, uninherited=0, effects=effects)
            )
            seen_times.add(time_ms)


def _refine_beat_length(times: list[float], coarse_beat_length: float) -> float | None:
    if len(times) < 8 or coarse_beat_length <= 0:
        return None
    t0 = times[0]
    span = times[-1] - t0
    if span <= 0:
        return None
    n_steps = max(8, round(span / coarse_beat_length))
    best_bl = coarse_beat_length
    best_score = float("inf")
    lo = coarse_beat_length * 0.95
    hi = coarse_beat_length * 1.05
    candidates = 4000
    for k in range(candidates + 1):
        bl = lo + (hi - lo) * k / candidates
        score = 0.0
        for t in times:
            idx_f = (t - t0) / bl
            residual = (idx_f - round(idx_f)) * bl
            score += residual * residual
        if score < best_score:
            best_score = score
            best_bl = bl
    if not 100.0 < best_bl < 2000.0:
        return None
    return best_bl


def _infer_beat_length_ms(
    groups: list[tuple[int, list[Event]]],
    cfg: TokenizerConfig,
) -> float | None:
    beat_times_ms: list[float] = []
    for abs_bin, group in groups:
        if any(ev.type in (EventType.BEAT, EventType.MEASURE) for ev in group):
            beat_times_ms.append(float(abs_bin * cfg.dt_bin_ms))
    if len(beat_times_ms) < 5:
        return None
    beat_times_ms.sort()
    deltas = [beat_times_ms[i + 1] - beat_times_ms[i] for i in range(len(beat_times_ms) - 1)]
    deltas.sort()
    median_delta = deltas[len(deltas) // 2]
    if median_delta < 100.0 or median_delta > 2000.0:
        return None
    refined = _refine_beat_length(beat_times_ms, median_delta)
    return refined if refined is not None else median_delta


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
    pending_marker: Event | None = None
    for ev in events:
        if ev.type in _OBJECT_MARKERS:
            pending_marker = ev
            continue
        if ev.type == EventType.ABS_TIME:
            if current_abs is not None:
                groups.append((current_abs, current))
            current_abs = ev.value
            current = []
            if pending_marker is not None:
                current.append(pending_marker)
                pending_marker = None
        else:
            current.append(ev)
    if current_abs is not None:
        groups.append((current_abs, current))
    return groups


def _find_marker(group: list[Event]) -> EventType | None:
    for ev in group:
        if ev.type in _OBJECT_MARKERS:
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


def _resnap_time_to_divisor(raw_time_ms: float, snap: int, timing_points: list[TimingPoint]) -> float:
    if snap <= 0:
        return raw_time_ms
    tp = _active_uninherited_tp(raw_time_ms, timing_points)
    if tp is None or tp.beat_length <= 0:
        return raw_time_ms
    tick_ms = tp.beat_length / snap
    offset = raw_time_ms - tp.time
    snapped_offset = round(offset / tick_ms) * tick_ms
    return tp.time + snapped_offset


def _active_uninherited_tp(time_ms: float, timing_points: list[TimingPoint]) -> TimingPoint | None:
    active: TimingPoint | None = None
    for tp in timing_points:
        if tp.uninherited != 1 or tp.beat_length <= 0:
            continue
        if tp.time <= time_ms + 1e-6:
            active = tp
        else:
            break
    if active is None:
        for tp in timing_points:
            if tp.uninherited == 1 and tp.beat_length > 0:
                return tp
    return active


def _find_spinner_end(
    group: list[Event],
    head_time_ms: float,
    timing_points: list[TimingPoint],
) -> float:
    duration_idx = _find_value(group, EventType.DURATION)
    if duration_idx is None:
        return head_time_ms + 1000.0
    beats = duration_index_to_beats(duration_idx)
    state = _state_at(head_time_ms, timing_points)
    return head_time_ms + beats * state.beat_length


def _build_slider(
    *,
    head_time_ms: float,
    head_pos: tuple[float, float],
    head_hit_sound: HitSound,
    head_sample: HitSample,
    new_combo: bool,
    group: list[Event],
    vocab: Vocab,
    slider_multiplier: float,
    timing_points: list[TimingPoint],
) -> Slider | None:
    segments, current_curve_type = _parse_anchor_chain(group, vocab)
    if not segments:
        return None

    slides = 1
    for ev in group:
        if ev.type == EventType.SLIDER_SLIDES:
            slides = max(1, min(8, ev.value))

    state = _state_at(head_time_ms, timing_points)
    geom_length = max(1.0, _total_length(head_pos, segments))
    denom = 100.0 * slider_multiplier * state.sv_multiplier
    duration_idx = _find_value(group, EventType.DURATION)
    if duration_idx is not None and denom > 0:
        beats = duration_index_to_beats(duration_idx)
        intended_duration_ms = beats * state.beat_length
        needed_length = intended_duration_ms * denom / (state.beat_length * max(1, slides))
        length = max(1.0, min(needed_length, geom_length * 1.5))
        duration_ms = length * state.beat_length * max(1, slides) / denom
    else:
        length = geom_length
        if denom > 0:
            duration_ms = length * state.beat_length * max(1, slides) / denom
        else:
            duration_ms = state.beat_length * max(1, slides)

    curves = [SliderCurve(curve_type=_infer_curve_type(seg, current_curve_type), curve_points=seg) for seg in segments]
    type_byte = 2 | (4 if new_combo else 0)
    params = SliderObjectParams(curves=curves, slides=slides, length=length, duration=duration_ms)
    return Slider(
        x=head_pos[0],
        y=head_pos[1],
        time=float(head_time_ms),
        type=type_byte,
        hit_sound=head_hit_sound,
        object_params=params,
        hit_sample=head_sample,
    )


def _parse_anchor_chain(
    group: list[Event], vocab: Vocab
) -> tuple[list[list[tuple[float, float]]], CurveType]:
    segments: list[list[tuple[float, float]]] = []
    current_points: list[tuple[float, float]] = []
    current_curve_type: CurveType = CurveType.LINEAR
    seen_head = False
    i = 0
    while i < len(group):
        ev = group[i]
        if not seen_head:
            if ev.type == EventType.SLIDER_HEAD:
                seen_head = True
            i += 1
            continue
        if ev.type == EventType.SLIDER_END:
            break
        if ev.type == EventType.RED_ANCHOR:
            if current_points:
                segments.append(current_points)
                current_points = []
            i += 1
            if i < len(group) and group[i].type == EventType.POS:
                current_points.append(vocab.grid.decode(group[i].value))
                i += 1
            continue
        if ev.type in _ANCHOR_TO_CURVE:
            new_type = _ANCHOR_TO_CURVE[ev.type]
            if new_type != current_curve_type and current_points:
                segments.append(current_points)
                current_points = []
            current_curve_type = new_type
            i += 1
            if i < len(group) and group[i].type == EventType.POS:
                current_points.append(vocab.grid.decode(group[i].value))
                i += 1
            continue
        i += 1
    if current_points:
        segments.append(current_points)
    return segments, current_curve_type


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
