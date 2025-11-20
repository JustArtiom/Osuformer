from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tokenizer import HitObjectTokenizer, TokenAttr, TokenType
from src.osu import Beatmap
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect discrete tokenization for a single osu! map.")
    parser.add_argument("--osu", required=True, help="Path to the .osu file to tokenize.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Which chunk (0-based) to inspect. Uses context/target/sample_hop beats from config.",
    )
    parser.add_argument(
        "--print-empty",
        action="store_true",
        help="If set, also print ticks that only contain padding.",
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


def _limit_token_count(tokens: List[List[int]], limit: int) -> List[List[int]]:
    trimmed: List[List[int]] = []
    idx = 0
    while idx < len(tokens) and len(trimmed) < limit:
        token = tokens[idx]
        token_type = token[TokenAttr.TYPE]
        if token_type == TokenType.SLIDER:
            group_end = idx + 1
            while group_end < len(tokens) and tokens[group_end][TokenAttr.TYPE] == TokenType.SLIDER_PATH:
                group_end += 1
            group_size = group_end - idx
            if len(trimmed) + group_size > limit:
                break
            trimmed.extend(tokens[idx:group_end])
            idx = group_end
            continue
        if token_type == TokenType.SLIDER_PATH:
            idx += 1
            continue
        trimmed.append(token)
        idx += 1
    return trimmed


def encode_chunk_tokens(
    beatmap: Beatmap,
    hit_objects: Sequence[object],
    chunk_start_ms: float,
    ticks_per_sample: int,
    tick_duration_ms: float,
    tokenizer: HitObjectTokenizer,
    tick_tolerance_ms: float,
) -> tuple[List[List[int]], int]:
    if not hit_objects:
        return [], 0
    tokens = tokenizer.tokenize(
        hit_objects,
        chunk_start_ms=chunk_start_ms,
        tick_duration_ms=tick_duration_ms,
        max_ticks=ticks_per_sample,
        slider_sv_lookup=lambda slider: effective_slider_sv(beatmap, float(getattr(slider, "time", 0.0))),
        tick_tolerance_ms=tick_tolerance_ms,
    )
    tokens = _limit_token_count(tokens, max(1, ticks_per_sample - 1))
    tokens.append(tokenizer.eos_token())
    token_length = min(len(tokens), ticks_per_sample)
    tokens = tokens[:token_length]
    if tokens and tokens[-1][TokenAttr.TYPE] != TokenType.EOS:
        tokens[-1] = tokenizer.eos_token()
    while len(tokens) < ticks_per_sample:
        tokens.append(tokenizer.pad_token())
    return tokens, token_length


def summarize_token(tokenizer: HitObjectTokenizer, token: Sequence[int]) -> str:
    ttype = token[TokenAttr.TYPE]
    if ttype == TokenType.EOS:
        return "EOS"
    tick = tokenizer.tick_from_id(token[TokenAttr.TICK])
    if ttype == TokenType.CIRCLE:
        x = tokenizer.coord_from_id(token[TokenAttr.X])
        y = tokenizer.coord_from_id(token[TokenAttr.Y])
        return f"CIRCLE tick={tick} pos=({x},{y})"
    if ttype == TokenType.SLIDER:
        duration = token[TokenAttr.DURATION] - 1
        slides = token[TokenAttr.SLIDES] - 1
        curve = token[TokenAttr.CURVE_TYPE]
        sv = token[TokenAttr.SLIDER_SV]
        x = tokenizer.coord_from_id(token[TokenAttr.X])
        y = tokenizer.coord_from_id(token[TokenAttr.Y])
        return (
            "SLIDER "
            f"tick={tick} pos=({x},{y}) "
            f"dur_ticks={duration} slides={slides} curve={curve} sv={sv}"
        )
    if ttype == TokenType.SLIDER_PATH:
        x = tokenizer.coord_from_id(token[TokenAttr.X])
        y = tokenizer.coord_from_id(token[TokenAttr.Y])
        return f"SLIDER_PATH tick={tick} pos=({x},{y})"
    return f"UNKNOWN type={ttype}"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    tokenizer = HitObjectTokenizer(data_cfg)

    osu_path = Path(args.osu).expanduser().resolve()
    beatmap = Beatmap(file_path=str(osu_path))

    bpm, offset = get_primary_bpm_and_offset(beatmap, data_cfg.get("default_bpm", 120.0))
    context_beats = data_cfg.get("context_beats", 8)
    target_beats = data_cfg.get("target_beats", 16)
    total_beats = context_beats + target_beats
    ticks_per_beat = data_cfg.get("ticks_per_beat", 4)
    sample_hop_beats = data_cfg.get("sample_hop_beats", target_beats)
    ticks_per_sample = total_beats * ticks_per_beat

    beat_duration_ms = 60000.0 / max(bpm, 1e-3)
    chunk_duration_ms = total_beats * beat_duration_ms
    sample_hop_ms = sample_hop_beats * beat_duration_ms
    tick_duration_ms = beat_duration_ms / ticks_per_beat

    chunk_index = max(0, args.chunk_index)
    chunk_start = max(0.0, offset) + chunk_index * sample_hop_ms
    chunk_end = chunk_start + chunk_duration_ms

    hit_objects = filter_hit_objects(getattr(beatmap, "hit_objects", []), chunk_start, chunk_end)
    tokens, token_length = encode_chunk_tokens(
        beatmap,
        hit_objects,
        chunk_start,
        ticks_per_sample,
        tick_duration_ms,
        tokenizer,
        tick_tolerance_ms=float(data_cfg.get("tick_tolerance_ms", 10.0)),
    )

    print(f"Chunk #{chunk_index} | start={chunk_start:.2f}ms end={chunk_end:.2f}ms | tokens={token_length}")
    for idx, token in enumerate(tokens):
        summary = summarize_token(tokenizer, token)
        if not summary and not args.print_empty:
            continue
        print(f"[{idx:03d}] {summary}")


if __name__ == "__main__":
    main()
