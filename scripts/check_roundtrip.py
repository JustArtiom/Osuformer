"""Round-trip test: real .osu file -> tokenize -> detokenize -> compare structure."""
from __future__ import annotations

import math
import sys
from pathlib import Path

from src.config.loader import load_config
from src.inference.detokenizer import events_to_beatmap
from src.osu.beatmap import Beatmap
from src.osu.hit_object import Circle, Slider, Spinner
from src.osu_tokenizer import (
    Vocab,
    beatmap_to_events,
    collect_timing_events,
    merge_by_time,
)


def summarize(beatmap: Beatmap) -> dict:
    n_circles = sum(1 for h in beatmap.hit_objects if isinstance(h, Circle))
    n_sliders = sum(1 for h in beatmap.hit_objects if isinstance(h, Slider))
    n_spinners = sum(1 for h in beatmap.hit_objects if isinstance(h, Spinner))
    times = [float(h.time) for h in beatmap.hit_objects]
    positions = [(float(h.x), float(h.y)) for h in beatmap.hit_objects]
    return {
        "circles": n_circles,
        "sliders": n_sliders,
        "spinners": n_spinners,
        "total": n_circles + n_sliders + n_spinners,
        "times": times,
        "positions": positions,
        "timing_points": len(beatmap.timing_points),
        "uninherited": sum(1 for tp in beatmap.timing_points if tp.uninherited == 1),
    }


def position_error(orig_positions: list, new_positions: list) -> dict:
    n = min(len(orig_positions), len(new_positions))
    if n == 0:
        return {"mean_px": 0.0, "max_px": 0.0}
    errors = [
        math.hypot(orig_positions[i][0] - new_positions[i][0], orig_positions[i][1] - new_positions[i][1])
        for i in range(n)
    ]
    return {"mean_px": sum(errors) / n, "max_px": max(errors), "compared": n}


def time_error(orig_times: list, new_times: list) -> dict:
    n = min(len(orig_times), len(new_times))
    if n == 0:
        return {"mean_ms": 0.0, "max_ms": 0.0}
    errors = [abs(orig_times[i] - new_times[i]) for i in range(n)]
    return {"mean_ms": sum(errors) / n, "max_ms": max(errors), "compared": n}


def filter_to_window(beatmap: Beatmap, window_start_ms: float, window_end_ms: float) -> Beatmap:
    """Return a copy with hit_objects restricted to the time window."""
    filtered_objs = [h for h in beatmap.hit_objects if window_start_ms <= float(h.time) < window_end_ms]
    return Beatmap(
        general=beatmap.general,
        editor=beatmap.editor,
        metadata=beatmap.metadata,
        difficulty=beatmap.difficulty,
        events=beatmap.events,
        timing_points=beatmap.timing_points,
        colours=beatmap.colours,
        hit_objects=filtered_objs,
    )


def main(osu_path: Path) -> None:
    cfg = load_config("config/config.yaml").tokenizer
    vocab = Vocab(cfg)
    window_total_ms = float(cfg.context_ms + cfg.generate_ms + cfg.lookahead_ms)

    print(f"loading {osu_path.name}...")
    full = Beatmap(file_path=str(osu_path))
    full_summary = summarize(full)
    print(
        f"  FULL MAP: {full_summary['total']} hit objects "
        f"({full_summary['circles']}c {full_summary['sliders']}s {full_summary['spinners']}sp), "
        f"{full_summary['timing_points']} timing points ({full_summary['uninherited']} uninherited)"
    )

    # Window aligned to the first uninherited timing point so detokenizer's
    # snap-to-divisor reconstructs identical times (otherwise window_start that
    # doesn't sit on a beat boundary causes systematic offset)
    tp_active = next((tp for tp in full.timing_points if tp.uninherited == 1 and tp.beat_length > 0), None)
    window_start = float(tp_active.time) if tp_active is not None else 0.0
    if full.hit_objects and tp_active is not None:
        # Find a window with lots of hit objects: try windows at integer beat offsets
        beat_ms = tp_active.beat_length
        candidate_starts = [tp_active.time + n * beat_ms for n in range(0, 10000, 4) if tp_active.time + n * beat_ms < full.hit_objects[-1].time]
        best_count = 0
        best_start = window_start
        for start in candidate_starts:
            count = sum(1 for h in full.hit_objects if start <= float(h.time) < start + window_total_ms)
            if count > best_count:
                best_count = count
                best_start = start
        window_start = best_start
    window_end = window_start + window_total_ms

    print(f"  testing window [{window_start:.0f}ms, {window_end:.0f}ms] ({window_total_ms/1000:.0f}s)")
    original = filter_to_window(full, window_start, window_end)
    orig_summary = summarize(original)
    print(
        f"  WINDOW:   {orig_summary['total']} hit objects "
        f"({orig_summary['circles']}c {orig_summary['sliders']}s {orig_summary['spinners']}sp)"
    )

    print("tokenizing...")
    stream = beatmap_to_events(original, window_start, vocab, cfg, clamp_abs_time=True)
    timing = collect_timing_events(original, window_start, vocab, cfg, clamp_abs_time=True)
    merged = merge_by_time(stream.events, timing)
    print(f"  events: {len(merged)} (hit-object stream={len(stream.events)}, timing stream={len(timing)})")

    type_order = [er.type for er in vocab.output_ranges] + [er.type for er in vocab.input_ranges]
    type_to_idx = {t: i for i, t in enumerate(type_order)}
    unknown = [e for e in merged if e.type not in type_to_idx]
    if unknown:
        print(f"  WARNING: {len(unknown)} unknown event types: {[e.type.value for e in unknown[:5]]}")
        sys.exit(1)
    encoded_ids = [vocab.encode_event(e) for e in merged]
    print(f"  encoded {len(encoded_ids)} tokens, max id {max(encoded_ids)}, vocab_out {vocab.vocab_size_out}")

    print("detokenizing...")
    primary_bpm = 120.0
    if original.timing_points:
        for tp in original.timing_points:
            if tp.uninherited == 1 and tp.beat_length > 0:
                primary_bpm = 60000.0 / tp.beat_length
                break
    rebuilt = events_to_beatmap(
        events=merged,
        vocab=vocab,
        tokenizer_cfg=cfg,
        audio_filename="audio.mp3",
        bpm=primary_bpm,
        title="rt",
        artist="rt",
        creator="rt",
        version="rt",
        circle_size=float(original.difficulty.circle_size),
        approach_rate=float(original.difficulty.approach_rate),
        overall_difficulty=float(original.difficulty.overall_difficulty),
        hp_drain_rate=float(original.difficulty.hp_drain_rate),
        slider_multiplier=float(original.difficulty.slider_multiplier),
        auto_timing=False,
    )
    new_summary = summarize(rebuilt)
    print(
        f"  REBUILT:  {new_summary['total']} hit objects "
        f"({new_summary['circles']}c {new_summary['sliders']}s {new_summary['spinners']}sp), "
        f"{new_summary['timing_points']} timing points"
    )

    print()
    print("=== STRUCTURE COMPARISON ===")
    print(f"  hit count diff: {new_summary['total'] - orig_summary['total']:+d} "
          f"(original {orig_summary['total']}, rebuilt {new_summary['total']})")
    print(f"  circle diff: {new_summary['circles'] - orig_summary['circles']:+d}")
    print(f"  slider diff: {new_summary['sliders'] - orig_summary['sliders']:+d}")
    print(f"  spinner diff: {new_summary['spinners'] - orig_summary['spinners']:+d}")

    pos_err = position_error(orig_summary["positions"], new_summary["positions"])
    orig_times_local = [t - window_start for t in orig_summary["times"]]
    time_err = time_error(orig_times_local, new_summary["times"])
    print()
    print("=== ACCURACY (paired by index, may misalign if counts differ) ===")
    print(f"  position error (px): mean {pos_err['mean_px']:.2f}, max {pos_err['max_px']:.2f}, n={pos_err['compared']}")
    print(f"  time error (ms):     mean {time_err['mean_ms']:.2f}, max {time_err['max_ms']:.2f}, n={time_err['compared']}")

    print()
    print("=== TYPE ALIGNMENT (first 30 hit objects) ===")
    n = min(30, len(original.hit_objects), len(rebuilt.hit_objects))
    mismatches = 0
    for i in range(n):
        orig_type = type(original.hit_objects[i]).__name__
        new_type = type(rebuilt.hit_objects[i]).__name__
        if orig_type != new_type:
            mismatches += 1
            if mismatches <= 5:
                print(f"  [{i}] type mismatch: {orig_type} -> {new_type}")
    if mismatches == 0:
        print(f"  all {n} types match")
    else:
        print(f"  {mismatches}/{n} type mismatches")

    sample_indices = [0, 5, 10, 20, 50, 100, 200] if len(original.hit_objects) >= 200 else [0, 1, 2, 5, 10]
    print()
    print("=== SAMPLE OBJECTS ===")
    for idx in sample_indices:
        if idx >= len(original.hit_objects) or idx >= len(rebuilt.hit_objects):
            continue
        o = original.hit_objects[idx]
        r = rebuilt.hit_objects[idx]
        print(
            f"  [{idx}] orig: {type(o).__name__} t={o.time:.0f} ({o.x:.0f},{o.y:.0f})  "
            f"rebuilt: {type(r).__name__} t={r.time:.0f} ({r.x:.0f},{r.y:.0f})"
        )

    print()
    print("=== SLIDER STRUCTURE COMPARISON ===")
    orig_sliders = [(i, h) for i, h in enumerate(original.hit_objects) if isinstance(h, Slider)]
    new_sliders = [(i, h) for i, h in enumerate(rebuilt.hit_objects) if isinstance(h, Slider)]
    if not orig_sliders:
        print("  no sliders in window")
    else:
        n_compare = min(5, len(orig_sliders), len(new_sliders))
        slides_diff = 0
        anchor_count_diff = 0
        for k in range(n_compare):
            i_o, o = orig_sliders[k]
            i_n, n = new_sliders[k]
            o_anchors = sum(len(c.curve_points) for c in o.object_params.curves)
            n_anchors = sum(len(c.curve_points) for c in n.object_params.curves)
            o_curves = [c.curve_type.value for c in o.object_params.curves]
            n_curves = [c.curve_type.value for c in n.object_params.curves]
            print(
                f"  [orig idx {i_o}] slides={o.object_params.slides} anchors={o_anchors} "
                f"curves={o_curves} length={o.object_params.length:.0f}"
            )
            print(
                f"  [new  idx {i_n}] slides={n.object_params.slides} anchors={n_anchors} "
                f"curves={n_curves} length={n.object_params.length:.0f}"
            )
            if o.object_params.slides != n.object_params.slides:
                slides_diff += 1
            if o_anchors != n_anchors:
                anchor_count_diff += 1
        print(f"  -- slides mismatches: {slides_diff}/{n_compare}, anchor count mismatches: {anchor_count_diff}/{n_compare}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
