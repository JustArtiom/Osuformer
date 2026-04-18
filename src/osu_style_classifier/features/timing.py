from __future__ import annotations

from src.osu.beatmap import Beatmap

from .feature_set import TimingFeatures


def extract_timing_features(beatmap: Beatmap) -> TimingFeatures:
    tps = sorted(beatmap.timing_points, key=lambda t: t.time)
    uninherited = [t for t in tps if t.uninherited == 1 and t.beat_length > 0]
    inherited = [t for t in tps if t.uninherited == 0]

    if not uninherited:
        return TimingFeatures()

    bpm_durations: dict[float, float] = {}
    end_time = _beatmap_end_time(beatmap)
    for i, tp in enumerate(uninherited):
        start = tp.time
        stop = uninherited[i + 1].time if i + 1 < len(uninherited) else end_time
        duration = max(0.0, stop - start)
        bpm = 60000.0 / tp.beat_length
        bpm_durations[bpm] = bpm_durations.get(bpm, 0.0) + duration

    primary_bpm = max(bpm_durations.items(), key=lambda kv: kv[1])[0] if bpm_durations else 0.0
    bpms = list(bpm_durations.keys())

    sv_mults = [-100.0 / tp.beat_length for tp in inherited if tp.beat_length < 0]
    total_minutes = max(1e-6, end_time / 60000.0)
    distinct_sv_changes = 0
    prev_mult = 1.0
    for tp in inherited:
        if tp.beat_length >= 0:
            continue
        mult = -100.0 / tp.beat_length
        if abs(mult - prev_mult) > 0.05:
            distinct_sv_changes += 1
            prev_mult = mult
    sv_changes_per_minute = distinct_sv_changes / total_minutes

    meters = [t.meter for t in uninherited]
    non_4_4 = sum(1 for m in meters if m != 4)

    return TimingFeatures(
        primary_bpm=primary_bpm,
        bpm_min=min(bpms),
        bpm_max=max(bpms),
        bpm_count=len(bpms),
        uninherited_count=len(uninherited),
        inherited_count=len(inherited),
        meters=meters,
        non_4_4_ratio=non_4_4 / len(meters) if meters else 0.0,
        sv_multipliers=sv_mults,
        sv_changes_per_minute=sv_changes_per_minute,
        avg_sv_multiplier=sum(sv_mults) / len(sv_mults) if sv_mults else 1.0,
        max_sv_multiplier=max(sv_mults) if sv_mults else 1.0,
        min_sv_multiplier=min(sv_mults) if sv_mults else 1.0,
    )


def _beatmap_end_time(beatmap: Beatmap) -> float:
    if not beatmap.hit_objects:
        return 0.0
    last = beatmap.hit_objects[-1]
    end = float(last.time)
    params = getattr(last, "object_params", None)
    if params is not None:
        duration = getattr(params, "duration", None)
        if duration is not None:
            end = max(end, float(last.time) + float(duration))
        end_time = getattr(params, "end_time", None)
        if end_time is not None:
            end = max(end, float(end_time))
    return end
