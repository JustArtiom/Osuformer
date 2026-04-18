from __future__ import annotations

from typing import Sequence

from src.osu.beatmap import Beatmap
from src.osu.hit_object import HitObject
from src.osu.timing_point import TimingPoint

from .feature_set import StreamFeatures
from .geometry import distance
from .snap import active_uninherited, classify_snap


def extract_stream_features(
    beatmap: Beatmap,
    snap_tolerance_ms: float,
    min_run_length: int,
    spaced_px_per_ms: float,
    cutstream_max_gap_beats: float,
    max_stream_interval_ms: float,
) -> StreamFeatures:
    objs = beatmap.hit_objects
    if len(objs) < min_run_length:
        return StreamFeatures()

    tps = sorted(beatmap.timing_points, key=lambda t: t.time)

    runs: list[list[int]] = []
    current: list[int] = []
    for i in range(len(objs) - 1):
        dt = float(objs[i + 1].time) - float(objs[i].time)
        tp = active_uninherited(tps, float(objs[i].time))
        if tp is None:
            if len(current) >= min_run_length:
                runs.append(current)
            current = []
            continue
        div = classify_snap(dt, tp.beat_length, snap_tolerance_ms)
        is_stream_tap = (div is not None and div >= 4) and dt <= max_stream_interval_ms
        if is_stream_tap:
            if not current:
                current.append(i)
            current.append(i + 1)
        else:
            if len(current) >= min_run_length:
                runs.append(current)
            current = []
    if len(current) >= min_run_length:
        runs.append(current)

    stream_runs = [len(r) for r in runs]
    max_run = max(stream_runs) if stream_runs else 0
    stream_objects = sum(stream_runs)
    stream_ratio = stream_objects / len(objs)

    burst_runs: list[list[int]] = []
    for i in range(len(objs) - 1):
        dt = float(objs[i + 1].time) - float(objs[i].time)
        tp = active_uninherited(tps, float(objs[i].time))
        if tp is None:
            continue
        div = classify_snap(dt, tp.beat_length, snap_tolerance_ms)
        is_stream_tap = (div is not None and div >= 4) and dt <= max_stream_interval_ms
        if is_stream_tap:
            if burst_runs and burst_runs[-1][-1] == i:
                burst_runs[-1].append(i + 1)
            else:
                burst_runs.append([i, i + 1])
    burst_count = sum(1 for r in burst_runs if 3 <= len(r) <= 9)

    spaced_distances: list[float] = []
    spaced_run_count = 0
    for r in runs:
        dists = [
            distance((float(objs[r[k]].x), float(objs[r[k]].y)), (float(objs[r[k + 1]].x), float(objs[r[k + 1]].y)))
            for k in range(len(r) - 1)
        ]
        spaced_distances.extend(dists)
        avg_dt = 0.0
        if len(r) > 1:
            total_dt = float(objs[r[-1]].time) - float(objs[r[0]].time)
            avg_dt = total_dt / max(1, len(r) - 1)
        avg_dist = sum(dists) / len(dists) if dists else 0.0
        if avg_dt > 0 and avg_dist / avg_dt >= spaced_px_per_ms:
            spaced_run_count += 1

    cutstream_count = _count_cutstreams(objs, tps, snap_tolerance_ms, cutstream_max_gap_beats, min_run_length // 2)

    longest_sustained = 0.0
    for r in runs:
        duration_ms = float(objs[r[-1]].time) - float(objs[r[0]].time)
        longest_sustained = max(longest_sustained, duration_ms / 1000.0)

    return StreamFeatures(
        stream_runs=stream_runs,
        stream_run_distances_px=spaced_distances,
        max_run_length=max_run,
        stream_object_ratio=stream_ratio,
        burst_count=burst_count,
        cutstream_count=cutstream_count,
        spaced_stream_count=spaced_run_count,
        longest_sustained_stream_seconds=longest_sustained,
    )


def _count_cutstreams(
    objs: Sequence[HitObject],
    tps: Sequence[TimingPoint],
    snap_tolerance_ms: float,
    cutstream_max_gap_beats: float,
    min_pre_run: int,
) -> int:
    count = 0
    run_len = 0
    for i in range(len(objs) - 1):
        dt = float(objs[i + 1].time) - float(objs[i].time)
        tp = active_uninherited(list(tps), float(objs[i].time))
        if tp is None:
            run_len = 0
            continue
        div = classify_snap(dt, tp.beat_length, snap_tolerance_ms)
        if div is not None and div >= 4:
            run_len += 1
        else:
            gap_beats = dt / tp.beat_length if tp.beat_length > 0 else 0.0
            if run_len >= min_pre_run and 0.25 < gap_beats <= cutstream_max_gap_beats:
                count += 1
            run_len = 0
    return count
