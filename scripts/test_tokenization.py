from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tokenizer import HitObjectTokenizer, TokenAttr, TokenType
from src.osu import Beatmap, Circle, Slider
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect discrete tokenization for a single osu! map.")
    parser.add_argument("--osu", required=True, help="Path to the .osu file to tokenize.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Which chunk (0-based) to inspect. Uses beats_per_sample/sample_hop_beats from config.",
    )
    parser.add_argument(
        "--print-empty",
        action="store_true",
        help="If set, also print ticks that contain the EMPTY token.",
    )
    return parser.parse_args()


def get_primary_bpm_and_offset(beatmap: Beatmap, default_bpm: float) -> tuple[float, float]:
    for tp in getattr(beatmap, "timing_points", []):
        if tp.uninherited == 1:
            bpm = tp.get_bpm()
            if bpm > 0:
                return bpm, tp.time
    return default_bpm, 0.0


def filter_hit_objects(
    hit_objects: Sequence[object],
    start_ms: float,
    end_ms: float,
) -> List[object]:
    return [ho for ho in hit_objects if start_ms <= float(getattr(ho, "time", 0.0)) < end_ms]


def effective_slider_sv(beatmap: Beatmap, time_ms: float) -> float:
    difficulty = getattr(beatmap, "difficulty", None)
    base_sv = float(getattr(difficulty, "slider_multiplier", 1.0))
    tp = beatmap.get_previous_timing_point(time_ms, filter=lambda t: t.uninherited == 0)
    sv_mult = tp.get_slider_velocity_multiplier() if tp else 1.0
    return max(1e-3, base_sv * sv_mult)


def encode_chunk_tokens(
    beatmap: Beatmap,
    hit_objects: Sequence[object],
    chunk_start_ms: float,
    ticks_per_sample: int,
    tick_duration_ms: float,
    tokenizer: HitObjectTokenizer,
    tick_tolerance_ms: float,
) -> tuple[List[List[int]], int]:
    events = sorted(hit_objects, key=lambda h: float(getattr(h, "time", 0.0)))
    tokens: List[List[int]] = []
    prev_time = chunk_start_ms
    tick_ms = max(1e-6, tick_duration_ms)
    emitted_event = False
    chunk_cutoff = chunk_start_ms + max(0, ticks_per_sample - 1) * tick_ms
    chunk_end_limit = chunk_start_ms + ticks_per_sample * tick_ms

    for ho in events:
        event_time = float(getattr(ho, "time", 0.0))
        if event_time >= chunk_cutoff:
            break
        delta_ms = max(0.0, event_time - prev_time)
        delta_ticks = int(round(delta_ms / tick_ms))
        delta_ticks = max(0, min(delta_ticks, tokenizer.max_delta_ticks))
        if emitted_event:
            delta_ticks = max(1, delta_ticks)
        snapped_time = prev_time + delta_ticks * tick_ms
        if abs(event_time - snapped_time) > tick_tolerance_ms:
            continue
        if snapped_time >= chunk_cutoff:
            break

        if isinstance(ho, Circle):
            token = tokenizer.encode_circle(float(ho.x), float(ho.y), delta_ticks)
        elif isinstance(ho, Slider):
            sv = effective_slider_sv(beatmap, event_time)
            token = tokenizer.encode_slider(ho, tick_duration_ms, sv, delta_ticks)
        else:
            continue

        if isinstance(ho, Slider):
            duration_ms = float(getattr(getattr(ho, "object_params", None), "duration", 0.0) or 0.0)
            duration_ticks = max(1, int(round(duration_ms / tick_ms)))
            if snapped_time + duration_ticks * tick_ms > chunk_end_limit:
                break
        else:
            duration_ticks = 0

        tokens.append(token)
        emitted_event = True
        if isinstance(ho, Slider):
            prev_time = snapped_time + duration_ticks * tick_ms
        else:
            prev_time = snapped_time

    tokens.append(tokenizer.eos_token())
    token_length = min(len(tokens), ticks_per_sample)
    tokens = tokens[:token_length]
    if tokens[-1][TokenAttr.TYPE] != TokenType.EOS:
        tokens[-1] = tokenizer.eos_token()
    while len(tokens) < ticks_per_sample:
        tokens.append(tokenizer.pad_token())
    return tokens, token_length


def decode_token(tokenizer: HitObjectTokenizer, token: Sequence[int]) -> dict:
    decoded = tokenizer.decode_token(token)
    return decoded


def summarize_token(token: Sequence[int]) -> str:
    ttype = token[TokenAttr.TYPE]
    if ttype == TokenType.EOS:
        return "EOS"
    if ttype == TokenType.CIRCLE:
        return f"CIRCLE Δ={token[TokenAttr.DELTA]} start=({token[TokenAttr.START_X]},{token[TokenAttr.START_Y]})"
    if ttype == TokenType.SLIDER:
        return (
            "SLIDER "
            f"Δ={token[TokenAttr.DELTA]} "
            f"start=({token[TokenAttr.START_X]},{token[TokenAttr.START_Y]}) "
            f"end=({token[TokenAttr.END_X]},{token[TokenAttr.END_Y]}) "
            f"dur={token[TokenAttr.DURATION]} slides={token[TokenAttr.SLIDES]} "
            f"sv={token[TokenAttr.SLIDER_SV]}"
        )
    return f"UNKNOWN type={ttype}"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    tokenizer = HitObjectTokenizer(data_cfg)

    osu_path = Path(args.osu).expanduser().resolve()
    beatmap = Beatmap(file_path=str(osu_path))

    bpm, offset = get_primary_bpm_and_offset(beatmap, data_cfg.get("default_bpm", 120.0))
    beats_per_sample = data_cfg.get("beats_per_sample", 16)
    ticks_per_beat = data_cfg.get("ticks_per_beat", 8)
    sample_hop_beats = data_cfg.get("sample_hop_beats", max(1, beats_per_sample // 2))
    ticks_per_sample = beats_per_sample * ticks_per_beat

    beat_duration_ms = 60000.0 / max(bpm, 1e-3)
    chunk_duration_ms = beats_per_sample * beat_duration_ms
    sample_hop_ms = sample_hop_beats * beat_duration_ms
    tick_duration_ms = beat_duration_ms / ticks_per_beat
    tick_tolerance_ms = float(data_cfg.get("tick_tolerance_ms", 3.0))

    start = max(0.0, offset) + args.chunk_index * sample_hop_ms
    end = start + chunk_duration_ms
    hit_objects = filter_hit_objects(getattr(beatmap, "hit_objects", []), start, end)

    print(f"[INFO] BPM={bpm:.2f}, offset={offset:.2f} ms")
    print(f"[INFO] Chunk {args.chunk_index}: start={start:.2f} ms, end={end:.2f} ms")
    print(f"[INFO] Hit objects in window: {len(hit_objects)} (total map: {len(getattr(beatmap, 'hit_objects', []))})")

    tokens, token_length = encode_chunk_tokens(
        beatmap, hit_objects, start, ticks_per_sample, tick_duration_ms, tokenizer, tick_tolerance_ms
    )
    decoded = [decode_token(tokenizer, token) for token in tokens[:token_length]]

    current_time = start
    for idx, (raw, dec) in enumerate(zip(tokens, decoded)):
        delta_ticks = dec.get("delta_ticks", 0)
        current_time += delta_ticks * tick_duration_ms
        summary = summarize_token(raw)
        print(f"[IDX {idx:03d} | t={current_time:.2f} ms] {summary} -> {dec}")
        if dec["type"] == TokenType.SLIDER:
            duration_ticks = dec.get("duration_ticks", 0)
            current_time += max(0, duration_ticks) * tick_duration_ms
        elif dec["type"] == TokenType.CIRCLE:
            current_time = float(current_time)
        if dec["type"] == TokenType.EOS and not args.print_empty:
            break


if __name__ == "__main__":
    main()
