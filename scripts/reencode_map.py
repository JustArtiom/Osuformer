from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tokenizer import HitObjectTokenizer, TokenAttr, TokenType
from src.osu import Beatmap, Circle, Slider
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize + detokenize an osu! map to inspect fidelity.")
    parser.add_argument("--osu", required=True, help="Path to the .osu file.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the reconstructed .osu (default: input path with .tokenized.osu).",
    )
    parser.add_argument("--print-empty", action="store_true", help="Print ticks with EMPTY tokens as well.")
    return parser.parse_args()


def get_primary_bpm_and_offset(beatmap: Beatmap, default_bpm: float) -> Tuple[float, float]:
    for tp in getattr(beatmap, "timing_points", []):
        if tp.uninherited == 1:
            bpm = tp.get_bpm()
            if bpm > 0:
                return bpm, tp.time
    return default_bpm, 0.0


def ensure_constant_bpm(beatmap: Beatmap) -> bool:
    bpm_value = None
    for tp in getattr(beatmap, "timing_points", []):
        if tp.uninherited != 1:
            continue
        bpm = tp.get_bpm()
        if bpm <= 0:
            continue
        if bpm_value is None:
            bpm_value = bpm
        elif abs(bpm - bpm_value) > 1e-3:
            return False
    return bpm_value is not None


def effective_slider_sv(beatmap: Beatmap, time_ms: float) -> float:
    difficulty = getattr(beatmap, "difficulty", None)
    base_sv = float(getattr(difficulty, "slider_multiplier", 1.0))
    tp = beatmap.get_previous_timing_point(time_ms, filter=lambda t: t.uninherited == 0)
    sv_mult = tp.get_slider_velocity_multiplier() if tp else 1.0
    return max(1e-3, base_sv * sv_mult)


def encode_tokens(
    beatmap: Beatmap,
    tokenizer: HitObjectTokenizer,
    base_time: float,
    ticks_per_sample: int,
    tick_duration_ms: float,
    tick_tolerance_ms: float,
) -> Tuple[List[List[int]], float, float]:
    hit_objects = sorted(getattr(beatmap, "hit_objects", []), key=lambda ho: float(getattr(ho, "time", 0.0)))
    if not hit_objects:
        return [], base_time, base_time
    min_time = min(float(getattr(ho, "time", 0.0)) for ho in hit_objects)
    max_time = max(float(getattr(ho, "time", 0.0)) for ho in hit_objects)

    total_ticks = max(int(np.ceil((max_time - base_time) / tick_duration_ms)) + 2, ticks_per_sample)

    tokens: List[List[int]] = [tokenizer.empty_token() for _ in range(total_ticks)]
    occupied = np.zeros(total_ticks, dtype=bool)

    for ho in hit_objects:
        event_time = float(getattr(ho, "time", 0.0))
        rel = (event_time - base_time) / max(1e-6, tick_duration_ms)
        tick_idx = int(round(rel))
        if tick_idx < 0 or tick_idx >= total_ticks:
            continue
        tick_time = base_time + tick_idx * tick_duration_ms
        if abs(event_time - tick_time) > tick_tolerance_ms:
            continue
        if occupied[tick_idx]:
            continue

        if isinstance(ho, Circle):
            token = tokenizer.encode_circle(float(ho.x), float(ho.y))
        elif isinstance(ho, Slider):
            sv = effective_slider_sv(beatmap, event_time)
            token = tokenizer.encode_slider(ho, tick_duration_ms, sv)
        else:
            continue

        tokens[tick_idx] = token
        occupied[tick_idx] = True

    return tokens, min_time, max_time


def _clamp_coord(value: float | None, limit: int) -> int | None:
    if value is None:
        return None
    return int(max(0, min(limit, round(value))))


def decode_tokens(
    tokens: Sequence[Sequence[int]],
    tokenizer: HitObjectTokenizer,
    base_time: float,
    tick_duration_ms: float,
    data_cfg: dict,
    suppress_overlaps: bool = True,
) -> List[dict]:
    events: List[dict] = []
    width = data_cfg["osu_width"]
    height = data_cfg["osu_height"]
    beat_duration_ms = tick_duration_ms * data_cfg.get("ticks_per_beat", 8)
    blocked_until = float("-inf")

    for idx, token in enumerate(tokens):
        decoded = tokenizer.decode_token(token)
        token_type = decoded["type"]
        if token_type == TokenType.EMPTY:
            continue
        start_x = _clamp_coord(decoded.get("start_x"), width)
        start_y = _clamp_coord(decoded.get("start_y"), height)
        if start_x is None or start_y is None:
            continue
        time_ms = int(round(base_time + idx * tick_duration_ms))
        if suppress_overlaps and time_ms < blocked_until:
            continue

        if token_type == TokenType.CIRCLE:
            events.append({"type": "circle", "time": time_ms, "x": start_x, "y": start_y, "end_time": time_ms})
            continue

        end_x = _clamp_coord(decoded.get("end_x"), width)
        end_y = _clamp_coord(decoded.get("end_y"), height)
        if end_x is None or end_y is None:
            continue

        points: List[Tuple[int, int]] = []
        ctrl1_x = _clamp_coord(decoded.get("ctrl1_x"), width)
        ctrl1_y = _clamp_coord(decoded.get("ctrl1_y"), height)
        ctrl2_x = _clamp_coord(decoded.get("ctrl2_x"), width)
        ctrl2_y = _clamp_coord(decoded.get("ctrl2_y"), height)
        if ctrl1_x is not None and ctrl1_y is not None:
            points.append((ctrl1_x, ctrl1_y))
        if ctrl2_x is not None and ctrl2_y is not None:
            points.append((ctrl2_x, ctrl2_y))
        points.append((end_x, end_y))

        duration_ticks = decoded.get("duration_ticks", 0)
        if duration_ticks <= 0:
            continue

        duration_ms = duration_ticks * tick_duration_ms
        slides = max(1, int(decoded.get("slides", 1)))
        sv_factor = decoded.get("sv_factor") or 1.0
        slider_length = duration_ms * sv_factor * 100.0 / (beat_duration_ms * slides)
        slider_length = max(5.0, slider_length)

        events.append(
            {
                "type": "slider",
                "time": time_ms,
                "x": start_x,
                "y": start_y,
                "curve_type": decoded.get("curve_type") or "L",
                "points": points,
                "slides": slides,
                "length": int(round(slider_length)),
                "sv_factor": float(sv_factor),
                "end_time": time_ms + int(round(duration_ms)),
            }
        )
        if suppress_overlaps:
            blocked_until = max(blocked_until, time_ms + duration_ms)

    events.sort(key=lambda e: e["time"])
    return events


def format_hitobjects(events: Sequence[dict]) -> List[str]:
    lines: List[str] = []
    for event in events:
        if event["type"] == "circle":
            lines.append(f"{event['x']},{event['y']},{event['time']},1,0,0:0:0:0:")
        elif event["type"] == "slider":
            points = "|".join(f"{x}:{y}" for x, y in event["points"])
            curve = f"{event['curve_type']}|{points}"
            lines.append(
                f"{event['x']},{event['y']},{event['time']},2,0,{curve},{event['slides']},{event['length']},0:0:0:0:"
            )
    return lines


def replace_section(text: str, section: str, lines: Sequence[str]) -> str:
    pattern = re.compile(rf"\[{section}\]\s*(.*?)(?=\n\[|\Z)", re.DOTALL)
    block = f"[{section}]\n" + "\n".join(lines) + "\n"
    if pattern.search(text):
        return pattern.sub(block, text, count=1)
    return text.rstrip() + "\n" + block


def normalize_slider_multiplier(text: str) -> str:
    return re.sub(r"(SliderMultiplier\s*:\s*)([0-9.]+)", r"\g<1>1", text, count=1)


def build_timing_points(offset_ms: float, beat_length: float, events: Sequence[dict]) -> List[str]:
    lines = [
        f"{offset_ms:.4f},{beat_length:.4f},4,2,0,50,1,0",
        f"{offset_ms:.4f},-100.0000,4,2,0,50,0,0",
    ]
    seen: set[Tuple[int, int]] = set()
    for event in events:
        if event.get("type") != "slider":
            continue
        sv = float(event.get("sv_factor") or 1.0)
        key = (int(round(event["time"])), int(round(sv * 1000)))
        if key in seen:
            continue
        seen.add(key)
        beat_len = -100.0 / max(sv, 1e-4)
        lines.append(f"{event['time']:.4f},{beat_len:.4f},4,2,0,50,0,0")
    return lines


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    tokenizer = HitObjectTokenizer(data_cfg)

    osu_path = Path(args.osu).expanduser().resolve()
    beatmap = Beatmap(file_path=str(osu_path))

    if not ensure_constant_bpm(beatmap):
        raise SystemExit("Map has variable BPM; tokenization test expects constant BPM.")

    bpm, offset = get_primary_bpm_and_offset(beatmap, data_cfg.get("default_bpm", 120.0))
    beat_duration_ms = 60000.0 / max(bpm, 1e-3)
    tick_duration_ms = beat_duration_ms / data_cfg.get("ticks_per_beat", 8)
    tick_tolerance_ms = float(data_cfg.get("tick_tolerance_ms", 3.0))
    ticks_per_sample = data_cfg.get("beats_per_sample", 16) * data_cfg.get("ticks_per_beat", 8)

    base_time = min(offset, min(float(getattr(ho, "time", 0.0)) for ho in getattr(beatmap, "hit_objects", []) or [0.0]))

    tokens, min_time, max_time = encode_tokens(
        beatmap,
        tokenizer,
        base_time=base_time,
        ticks_per_sample=ticks_per_sample,
        tick_duration_ms=tick_duration_ms,
        tick_tolerance_ms=tick_tolerance_ms,
    )

    print(f"[INFO] BPM={bpm:.2f}, offset={offset:.2f} ms, base_time={base_time:.2f} ms")
    print(f"[INFO] HitObjects: {len(getattr(beatmap, 'hit_objects', []))} | Tokens produced: {len(tokens)}")
    print(f"[INFO] Time span: {min_time:.2f} → {max_time:.2f} ms")

    if args.print_empty:
        for idx, token in enumerate(tokens[: min(20_000, len(tokens))]):
            decoded = tokenizer.decode_token(token)
            print(f"TICK {idx:05d}: {decoded}")

    suppress_overlaps = bool(data_cfg.get("suppress_overlap", True))

    events = decode_tokens(
        tokens,
        tokenizer,
        base_time,
        tick_duration_ms,
        data_cfg,
        suppress_overlaps=suppress_overlaps,
    )
    hitobject_lines = format_hitobjects(events)

    template_text = osu_path.read_text(encoding="utf-8")
    beat_length = 60000.0 / max(bpm, 1e-3)
    timing_lines = build_timing_points(offset, beat_length, events)
    output_text = normalize_slider_multiplier(template_text)
    output_text = replace_section(output_text, "TimingPoints", timing_lines)
    output_text = replace_section(output_text, "HitObjects", hitobject_lines)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = (ROOT / f"{osu_path.stem}.tokenized.osu").resolve()
    output_path.write_text(output_text, encoding="utf-8")
    print(f"[INFO] Wrote detokenized map → {output_path}")


if __name__ == "__main__":
    main()
