from __future__ import annotations

from typing import Sequence

from src.osu.beatmap import Beatmap, HitObjectType
from src.osu.hit_object import Circle, HitObject, Slider, Spinner

from .feature_set import ObjectFeatures, SnapFeatures
from .geometry import distance, entropy_bits, flow_angle_deg, mean, stddev
from .snap import active_uninherited, classify_snap


def extract_object_features(beatmap: Beatmap, snap_tolerance_ms: float) -> tuple[ObjectFeatures, SnapFeatures]:
    objs = beatmap.hit_objects
    if not objs:
        return ObjectFeatures(), SnapFeatures()

    circles = sum(1 for o in objs if isinstance(o, Circle))
    sliders = sum(1 for o in objs if isinstance(o, Slider))
    spinners = sum(1 for o in objs if isinstance(o, Spinner))

    spinner_durations = [
        float(o.object_params.end_time) - float(o.time)
        for o in objs
        if isinstance(o, Spinner)
    ]

    start_times = [float(o.time) for o in objs]
    end_times = [_object_end_time(o) for o in objs]
    total_length_ms = max(end_times) - min(start_times) if objs else 0.0
    total_length_seconds = total_length_ms / 1000.0

    drain_time_ms = 0.0
    for i in range(len(objs) - 1):
        gap = float(objs[i + 1].time) - float(objs[i].time)
        if 0 < gap <= 5000.0:
            drain_time_ms += gap
    drain_time_seconds = drain_time_ms / 1000.0

    density = len(objs) / total_length_seconds if total_length_seconds > 0 else 0.0
    peak_density = _peak_density(start_times, window_seconds=4.0)

    tps = sorted(beatmap.timing_points, key=lambda t: t.time)
    divisor_counts: dict[int, int] = {}
    unsnapped = 0
    total_intervals = 0
    for i in range(len(objs) - 1):
        dt = float(objs[i + 1].time) - float(objs[i].time)
        if dt <= 0:
            continue
        total_intervals += 1
        tp = active_uninherited(tps, float(objs[i].time))
        if tp is None:
            unsnapped += 1
            continue
        div = classify_snap(dt, tp.beat_length, snap_tolerance_ms)
        if div is None:
            unsnapped += 1
        else:
            divisor_counts[div] = divisor_counts.get(div, 0) + 1

    snap = SnapFeatures(
        divisor_counts=divisor_counts,
        unsnapped_count=unsnapped,
        total_intervals=total_intervals,
        diversity=len(divisor_counts),
        entropy_bits=entropy_bits(list(divisor_counts.values())),
        complex_divisor_count=sum(c for d, c in divisor_counts.items() if d in (5, 7, 9, 16)),
    )

    distances: list[float] = []
    flow_angles: list[float] = []
    for i in range(len(objs) - 1):
        a = (float(objs[i].x), float(objs[i].y))
        b = (float(objs[i + 1].x), float(objs[i + 1].y))
        distances.append(distance(a, b))
        if i + 2 < len(objs):
            c = (float(objs[i + 2].x), float(objs[i + 2].y))
            flow_angles.append(flow_angle_deg(a, b, c))

    simultaneous = _count_simultaneous(objs)

    features = ObjectFeatures(
        total=len(objs),
        circles=circles,
        sliders=sliders,
        spinners=spinners,
        spinner_durations_ms=spinner_durations,
        object_density_per_second=density,
        peak_density_per_second=peak_density,
        drain_time_seconds=drain_time_seconds,
        total_length_seconds=total_length_seconds,
        distances_px=distances,
        flow_angles_deg=flow_angles,
        avg_distance_px=mean(distances),
        std_distance_px=stddev(distances),
        avg_flow_angle_deg=mean(flow_angles),
        std_flow_angle_deg=stddev(flow_angles),
        simultaneous_object_count=simultaneous,
    )
    return features, snap


def _object_end_time(obj: HitObjectType) -> float:
    if isinstance(obj, Slider):
        return float(obj.time) + float(obj.object_params.duration)
    if isinstance(obj, Spinner):
        return float(obj.object_params.end_time)
    params = getattr(obj, "object_params", None)
    if params is not None:
        end = getattr(params, "end_time", None)
        if end is not None:
            return float(end)
    return float(obj.time)


def _peak_density(start_times: list[float], window_seconds: float) -> float:
    if not start_times:
        return 0.0
    window_ms = window_seconds * 1000.0
    best = 0
    j = 0
    for i in range(len(start_times)):
        while j < len(start_times) and start_times[j] - start_times[i] < window_ms:
            j += 1
        best = max(best, j - i)
    return best / window_seconds


def _count_simultaneous(objs: Sequence[HitObject]) -> int:
    count = 0
    for i in range(len(objs) - 1):
        a = objs[i]
        b = objs[i + 1]
        ta = float(getattr(a, "time"))
        tb = float(getattr(b, "time"))
        a_end = ta
        a_params = getattr(a, "object_params", None)
        if a_params is not None:
            dur = getattr(a_params, "duration", None)
            if dur is not None:
                a_end = ta + float(dur)
            end = getattr(a_params, "end_time", None)
            if end is not None:
                a_end = max(a_end, float(end))
        if tb < a_end - 1.0:
            count += 1
    return count
