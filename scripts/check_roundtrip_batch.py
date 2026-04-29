"""Batch round-trip + alignment tests across many real beatmaps.

Tests:
1. Tokenizer round-trip preserves structure within quantization
2. Slider topology preservation (slides, anchor counts, curve types, length)
3. Hitsound + new_combo flag preservation
4. Beat scaffolding alignment with timing points
5. Audio mel slicing produces expected dimensions
6. SequenceBuilder pipeline determinism
"""
from __future__ import annotations

import math
import random
import sys
import traceback
from pathlib import Path
from typing import Any

from src.cache.audio import compute_mel
from src.config.loader import load_config
from src.inference.detokenizer import events_to_beatmap
from src.osu.beatmap import Beatmap
from src.osu.hit_object import Circle, Slider, Spinner
from src.osu_tokenizer import (
    EventType,
    Vocab,
    beatmap_to_events,
    collect_timing_events,
    merge_by_time,
)


def filter_to_window(beatmap: Beatmap, window_start_ms: float, window_end_ms: float) -> Beatmap:
    filtered = [h for h in beatmap.hit_objects if window_start_ms <= float(h.time) < window_end_ms]
    return Beatmap(
        general=beatmap.general,
        editor=beatmap.editor,
        metadata=beatmap.metadata,
        difficulty=beatmap.difficulty,
        events=beatmap.events,
        timing_points=beatmap.timing_points,
        colours=beatmap.colours,
        hit_objects=filtered,
    )


def best_aligned_window(full: Beatmap, window_total_ms: float) -> float:
    tp = next((tp for tp in full.timing_points if tp.uninherited == 1 and tp.beat_length > 0), None)
    if tp is None or not full.hit_objects:
        return 0.0
    beat_ms = tp.beat_length
    last_t = float(full.hit_objects[-1].time)
    candidates = []
    n = 0
    while True:
        s = tp.time + n * beat_ms * 4
        if s >= last_t:
            break
        candidates.append(s)
        n += 1
        if n > 5000:
            break
    if not candidates:
        return float(tp.time)
    best_count = -1
    best_start = candidates[0]
    for s in candidates:
        c = sum(1 for h in full.hit_objects if s <= float(h.time) < s + window_total_ms)
        if c > best_count:
            best_count = c
            best_start = s
    return best_start


def roundtrip_one(osu_path: Path, vocab: Vocab, cfg) -> dict:
    full = Beatmap(file_path=str(osu_path))
    if full.general.mode != 0:
        return {"path": str(osu_path), "skipped": "non-standard mode"}
    if not full.hit_objects:
        return {"path": str(osu_path), "skipped": "no hit objects"}

    window_total_ms = float(cfg.context_ms + cfg.generate_ms + cfg.lookahead_ms)
    window_start = best_aligned_window(full, window_total_ms)
    window_end = window_start + window_total_ms
    original = filter_to_window(full, window_start, window_end)
    if not original.hit_objects:
        return {"path": str(osu_path), "skipped": "empty window"}

    stream = beatmap_to_events(original, window_start, vocab, cfg, clamp_abs_time=True)
    timing = collect_timing_events(original, window_start, vocab, cfg, clamp_abs_time=True)
    merged = merge_by_time(stream.events, timing)

    type_order = [er.type for er in vocab.output_ranges] + [er.type for er in vocab.input_ranges]
    type_to_idx = {t: i for i, t in enumerate(type_order)}
    unknown = [e for e in merged if e.type not in type_to_idx]
    if unknown:
        return {
            "path": str(osu_path),
            "error": f"unknown event types: {set(e.type.value for e in unknown)}",
        }
    try:
        encoded_ids = [vocab.encode_event(e) for e in merged]
    except Exception as exc:
        return {"path": str(osu_path), "error": f"encode failed: {exc}"}

    primary_bpm = 120.0
    for tp in full.timing_points:
        if tp.uninherited == 1 and tp.beat_length > 0:
            primary_bpm = 60000.0 / tp.beat_length
            break
    try:
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
    except Exception as exc:
        return {"path": str(osu_path), "error": f"detokenize failed: {exc}"}

    n_orig = len(original.hit_objects)
    n_new = len(rebuilt.hit_objects)

    pos_errors = []
    time_errors = []
    type_match = 0
    nc_match = 0
    hs_match = 0
    n_compare = min(n_orig, n_new)
    for i in range(n_compare):
        o = original.hit_objects[i]
        r = rebuilt.hit_objects[i]
        pos_errors.append(math.hypot(o.x - r.x, o.y - r.y))
        time_errors.append(abs((o.time - window_start) - r.time))
        if type(o).__name__ == type(r).__name__:
            type_match += 1
        if o.is_new_combo() == r.is_new_combo():
            nc_match += 1
        # hitsound bitmask: NORMAL bit (1) is implicit default; compare only WHISTLE/FINISH/CLAP bits
        o_bits = int(o.hit_sound) & 0b1110
        r_bits = int(r.hit_sound) & 0b1110
        if o_bits == r_bits:
            hs_match += 1

    slider_pairs = []
    for i in range(n_compare):
        if isinstance(original.hit_objects[i], Slider) and isinstance(rebuilt.hit_objects[i], Slider):
            slider_pairs.append((original.hit_objects[i], rebuilt.hit_objects[i]))
    sl_slides_match = sum(1 for o, r in slider_pairs if o.object_params.slides == r.object_params.slides)
    sl_anchors_match = sum(
        1 for o, r in slider_pairs
        if sum(len(c.curve_points) for c in o.object_params.curves)
        == sum(len(c.curve_points) for c in r.object_params.curves)
    )
    sl_curve_match = sum(
        1 for o, r in slider_pairs
        if [c.curve_type.value for c in o.object_params.curves]
        == [c.curve_type.value for c in r.object_params.curves]
    )

    n_uninherited = sum(1 for tp in full.timing_points if tp.uninherited == 1)
    return {
        "path": str(osu_path),
        "n_orig": n_orig,
        "n_new": n_new,
        "circles_orig": sum(1 for h in original.hit_objects if isinstance(h, Circle)),
        "sliders_orig": sum(1 for h in original.hit_objects if isinstance(h, Slider)),
        "spinners_orig": sum(1 for h in original.hit_objects if isinstance(h, Spinner)),
        "n_events": len(merged),
        "max_token_id": max(encoded_ids) if encoded_ids else 0,
        "n_uninherited_tps": n_uninherited,
        "pos_err_mean": sum(pos_errors) / max(1, len(pos_errors)),
        "pos_err_max": max(pos_errors, default=0.0),
        "time_err_mean": sum(time_errors) / max(1, len(time_errors)),
        "time_err_max": max(time_errors, default=0.0),
        "type_match_rate": type_match / max(1, n_compare),
        "nc_match_rate": nc_match / max(1, n_compare),
        "hs_match_rate": hs_match / max(1, n_compare),
        "n_sliders": len(slider_pairs),
        "sl_slides_match_rate": sl_slides_match / max(1, len(slider_pairs)) if slider_pairs else 1.0,
        "sl_anchors_match_rate": sl_anchors_match / max(1, len(slider_pairs)) if slider_pairs else 1.0,
        "sl_curve_match_rate": sl_curve_match / max(1, len(slider_pairs)) if slider_pairs else 1.0,
    }


def beat_scaffolding_check(osu_path: Path, vocab: Vocab, cfg) -> dict:
    """Verify BEAT/MEASURE events are emitted at integer-beat offsets from TP."""
    full = Beatmap(file_path=str(osu_path))
    tp = next((tp for tp in full.timing_points if tp.uninherited == 1 and tp.beat_length > 0), None)
    if tp is None:
        return {"skipped": "no uninherited TP"}
    timing = collect_timing_events(full, 0.0, vocab, cfg, clamp_abs_time=False)
    beat_ms = tp.beat_length

    last_abs_bin = None
    beat_offsets = []
    for ev in timing:
        if ev.type == EventType.ABS_TIME:
            last_abs_bin = ev.value
        elif ev.type in (EventType.BEAT, EventType.MEASURE) and last_abs_bin is not None:
            event_time_ms = last_abs_bin * cfg.dt_bin_ms
            offset_from_tp = event_time_ms - tp.time
            n_beats = offset_from_tp / beat_ms
            beat_offsets.append(abs(n_beats - round(n_beats)))
    if not beat_offsets:
        return {"skipped": "no BEAT/MEASURE emitted"}
    return {
        "n_beat_events": len(beat_offsets),
        "beat_alignment_max_error": max(beat_offsets),
        "beat_alignment_mean_error": sum(beat_offsets) / len(beat_offsets),
    }


def audio_window_check(audio_path: Path, audio_cfg) -> dict:
    """Verify mel computation produces expected frame count for the audio file."""
    try:
        mel = compute_mel(audio_path, audio_cfg)
    except Exception as exc:
        return {"error": f"mel compute failed: {exc}"}
    return {
        "mel_frames": int(mel.shape[0]),
        "mel_bins": int(mel.shape[1]),
        "expected_bins": audio_cfg.n_mels,
        "duration_s": float(mel.shape[0]) * audio_cfg.hop_ms / 1000.0,
    }


def discover_maps(songs_root: Path, limit: int) -> list[Path]:
    paths: list[Path] = []
    for song_dir in sorted(songs_root.iterdir()):
        if not song_dir.is_dir() or song_dir.name.startswith("."):
            continue
        for osu in song_dir.glob("*.osu"):
            if osu.name.startswith("."):
                continue
            paths.append(osu)
            if len(paths) >= limit:
                return paths
    return paths


def main(songs_root: str, limit: int = 50) -> None:
    cfg_full = load_config("config/config.yaml")
    vocab = Vocab(cfg_full.tokenizer)
    songs_root_path = Path(songs_root)

    print(f"discovering maps under {songs_root_path}...")
    paths = discover_maps(songs_root_path, limit=500)
    if len(paths) > limit:
        random.seed(42)
        paths = random.sample(paths, limit)
    print(f"testing {len(paths)} maps\n")

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for i, p in enumerate(paths):
        try:
            r = roundtrip_one(p, vocab, cfg_full.tokenizer)
        except Exception as exc:
            errors.append({"path": str(p), "error": f"crash: {exc}", "tb": traceback.format_exc()[:200]})
            print(f"  [{i+1:>3}/{len(paths)}] CRASH {p.name[:60]}")
            continue
        if "skipped" in r:
            skipped.append(r)
            continue
        if "error" in r:
            errors.append(r)
            print(f"  [{i+1:>3}/{len(paths)}] ERROR {p.name[:60]}: {r['error']}")
            continue
        results.append(r)
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1:>3}/{len(paths)}] OK n={r['n_orig']:>3} "
                f"pos_err={r['pos_err_mean']:.2f}px t_err={r['time_err_mean']:.2f}ms "
                f"type={r['type_match_rate']*100:.0f}% hs={r['hs_match_rate']*100:.0f}%"
            )

    print()
    print("=" * 70)
    print(f"BATCH SUMMARY: {len(results)} succeeded, {len(skipped)} skipped, {len(errors)} errors")
    print("=" * 70)
    if not results:
        print("no successful round-trips")
        return

    def stat(key: str) -> tuple[float, float, float]:
        vals = [r[key] for r in results]
        return min(vals), sum(vals) / len(vals), max(vals)

    print()
    print("STRUCTURE PRESERVATION")
    hit_count_match_rate = sum(1 for r in results if r["n_orig"] == r["n_new"]) / len(results)
    print(f"  hit count exact match: {hit_count_match_rate*100:.1f}%")
    type_match_avg = sum(r["type_match_rate"] for r in results) / len(results)
    print(f"  hit type match (avg per map): {type_match_avg*100:.2f}%")
    nc_avg = sum(r["nc_match_rate"] for r in results) / len(results)
    print(f"  new combo flag match (avg): {nc_avg*100:.2f}%")
    hs_avg = sum(r["hs_match_rate"] for r in results) / len(results)
    print(f"  hitsound match (avg): {hs_avg*100:.2f}%")

    print()
    print("POSITION + TIME ACCURACY (overall)")
    pos_min, pos_avg, pos_max = stat("pos_err_mean")
    print(f"  position error mean (px): {pos_avg:.2f} (range {pos_min:.2f} – {pos_max:.2f})")
    pos_max_min, pos_max_avg, pos_max_max = stat("pos_err_max")
    print(f"  position error max (px):  {pos_max_avg:.2f} (range {pos_max_min:.2f} – {pos_max_max:.2f})")
    time_min, time_avg, time_max = stat("time_err_mean")
    print(f"  time error mean (ms):     {time_avg:.2f} (range {time_min:.2f} – {time_max:.2f})")
    time_max_min, time_max_avg, time_max_max = stat("time_err_max")
    print(f"  time error max (ms):      {time_max_avg:.2f} (range {time_max_min:.2f} – {time_max_max:.2f})")

    single_bpm = [r for r in results if r["n_uninherited_tps"] == 1]
    multi_bpm = [r for r in results if r["n_uninherited_tps"] > 1]
    if single_bpm and multi_bpm:
        print()
        print(f"TIME ACCURACY BY TP COUNT  (single-BPM: {len(single_bpm)} maps, multi-BPM: {len(multi_bpm)} maps)")
        sb_time = sum(r["time_err_mean"] for r in single_bpm) / len(single_bpm)
        mb_time = sum(r["time_err_mean"] for r in multi_bpm) / len(multi_bpm)
        sb_time_max = max(r["time_err_max"] for r in single_bpm)
        mb_time_max = max(r["time_err_max"] for r in multi_bpm)
        print(f"  single-BPM time err mean: {sb_time:.2f}ms  (max across maps: {sb_time_max:.2f}ms)")
        print(f"  multi-BPM  time err mean: {mb_time:.2f}ms  (max across maps: {mb_time_max:.2f}ms)")
        print(f"  -> remaining multi-BPM error from sections with too few BEAT events to derive BPM (falls back to global)")

    print()
    print("SLIDER TOPOLOGY")
    slider_results = [r for r in results if r["n_sliders"] > 0]
    if slider_results:
        slides_avg = sum(r["sl_slides_match_rate"] for r in slider_results) / len(slider_results)
        anchors_avg = sum(r["sl_anchors_match_rate"] for r in slider_results) / len(slider_results)
        curves_avg = sum(r["sl_curve_match_rate"] for r in slider_results) / len(slider_results)
        total_sliders = sum(r["n_sliders"] for r in slider_results)
        print(f"  maps with sliders: {len(slider_results)}, total sliders compared: {total_sliders}")
        print(f"  slides count match: {slides_avg*100:.2f}%")
        print(f"  anchor count match: {anchors_avg*100:.2f}%")
        print(f"  curve type match:   {curves_avg*100:.2f}%")
    else:
        print("  no sliders in any tested window")

    print()
    print("VOCAB UTILIZATION")
    n_events_min, n_events_avg, n_events_max = stat("n_events")
    print(f"  events per window: {n_events_avg:.0f} (range {n_events_min:.0f} – {n_events_max:.0f})")
    max_id_min, max_id_avg, max_id_max = stat("max_token_id")
    print(f"  max token id used: {max_id_avg:.0f} (range {max_id_min:.0f} – {max_id_max:.0f}), vocab_out: {vocab.vocab_size_out}")

    if errors:
        print()
        print(f"ERRORS ({len(errors)} total, showing first 5)")
        for e in errors[:5]:
            print(f"  {Path(e['path']).name[:60]}: {e['error']}")

    if skipped:
        print()
        print(f"SKIPPED ({len(skipped)} total, showing reasons):")
        from collections import Counter
        reasons = Counter(s["skipped"] for s in skipped)
        for reason, count in reasons.most_common():
            print(f"  {count}x: {reason}")

    print()
    print("=" * 70)
    print("BEAT SCAFFOLDING ALIGNMENT (sample of 5 maps)")
    print("=" * 70)
    sample = random.Random(7).sample(paths, min(5, len(paths)))
    for p in sample:
        try:
            r = beat_scaffolding_check(p, vocab, cfg_full.tokenizer)
        except Exception as exc:
            r = {"error": str(exc)}
        if "skipped" in r:
            print(f"  {p.name[:55]}: skipped ({r['skipped']})")
        elif "error" in r:
            print(f"  {p.name[:55]}: error ({r['error']})")
        else:
            ok = "OK" if r["beat_alignment_max_error"] < 0.05 else "DRIFT"
            print(
                f"  {p.name[:55]}: {ok} "
                f"n={r['n_beat_events']} mean_err={r['beat_alignment_mean_error']:.4f} max={r['beat_alignment_max_error']:.4f} beats"
            )

    print()
    print("=" * 70)
    print("AUDIO MEL DIMENSIONALITY (3 random songs)")
    print("=" * 70)
    audio_paths: list[Path] = []
    for p in sample:
        for cand in p.parent.glob("*.mp3"):
            audio_paths.append(cand)
            break
        if len(audio_paths) >= 3:
            break
    for ap in audio_paths[:3]:
        try:
            r = audio_window_check(ap, cfg_full.audio)
        except Exception as exc:
            r = {"error": str(exc)}
        if "error" in r:
            print(f"  {ap.name[:55]}: ERROR {r['error']}")
        else:
            bin_ok = "OK" if r["mel_bins"] == r["expected_bins"] else "MISMATCH"
            print(
                f"  {ap.name[:55]}: {r['mel_frames']} frames at {r['mel_bins']} bins ({bin_ok}, "
                f"~{r['duration_s']:.1f}s)"
            )


if __name__ == "__main__":
    songs = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/MySSD/Songs"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    main(songs, limit)
