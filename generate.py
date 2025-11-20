from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import math
import numpy as np
import torch

from src.audio import MelSpec, prepare_audio
from src.models import ConformerSeq2Seq
from src.osu import Beatmap
from src.utils.config import load_config
from src.data.tokenizer import HitObjectTokenizer, TokenAttr, TokenType


@dataclass
class AudioChunk:
    audio: torch.Tensor
    mask: torch.Tensor
    start_ms: float
    tick_duration_ms: float
    total_ticks: int
    context_ticks: int
    target_ticks: int
    beat_duration_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an osu! beatmap from audio using a trained model.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file to convert.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Model checkpoint to load.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file used for training.")
    parser.add_argument("--output", type=str, default=None, help="Where to save the generated .osu file.")
    parser.add_argument("--template", type=str, default=None, help="Optional template .osu file to preserve metadata.")
    parser.add_argument("--bpm", type=float, default=None, help="Override BPM for beat alignment.")
    parser.add_argument("--offset", type=float, default=None, help="Override offset (ms) for beat alignment.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run generation on.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for coordinate/EOS logits (higher = more random).",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        name = "cpu"
    return torch.device(name)


def load_or_create_mel(audio_path: Path, audio_cfg: dict) -> MelSpec:
    npz_path = Path(str(audio_path) + ".mel.npz")
    if not npz_path.exists():
        prepare_audio(
            str(audio_path),
            audio_cfg["sample_rate"],
            audio_cfg["hop_ms"],
            audio_cfg["win_ms"],
            audio_cfg["n_mels"],
            audio_cfg["n_fft"],
            force=False,
        )
    return MelSpec.load_npz(str(npz_path))


def get_timing_from_template(template_path: Path) -> Tuple[float, float]:
    beatmap = Beatmap(file_path=str(template_path))
    bpm = 0.0
    offset = 0.0
    for tp in getattr(beatmap, "timing_points", []):
        if tp.uninherited == 1:
            bpm = tp.get_bpm()
            offset = tp.time
            break
    return bpm, offset


def build_chunks(spec: MelSpec, data_cfg: dict, audio_cfg: dict, bpm: float, offset_ms: float) -> List[AudioChunk]:
    ticks_per_beat = int(data_cfg.get("ticks_per_beat", 4))
    context_beats = int(data_cfg.get("context_beats", 8))
    target_beats = int(data_cfg.get("target_beats", 16))
    total_beats = context_beats + target_beats
    sample_hop_beats = int(data_cfg.get("sample_hop_beats", target_beats))
    sample_hop_beats = max(1, sample_hop_beats)

    total_ticks = total_beats * ticks_per_beat
    context_ticks = context_beats * ticks_per_beat
    target_ticks = target_beats * ticks_per_beat

    beat_duration_ms = 60000.0 / max(bpm, 1e-3)
    chunk_duration_ms = total_beats * beat_duration_ms
    sample_hop_ms = sample_hop_beats * beat_duration_ms
    tick_duration_ms = beat_duration_ms / ticks_per_beat
    frames_per_chunk = max(1, int(round(chunk_duration_ms / spec.frame_duration_ms)))
    raw_frames_per_chunk = frames_per_chunk
    frame_stride = 1
    max_frames = audio_cfg.get("max_frames_per_sample")
    if max_frames and max_frames > 0 and raw_frames_per_chunk > max_frames:
        frame_stride = int(math.ceil(raw_frames_per_chunk / max_frames))
        frames_per_chunk = int(math.ceil(raw_frames_per_chunk / frame_stride))
        print(
            f"[INFO] Downsampling generation window: {raw_frames_per_chunk} frames -> "
            f"{frames_per_chunk} (stride {frame_stride})"
        )

    normalize_audio = bool(audio_cfg.get("normalize_audio", True))
    if normalize_audio:
        mean = torch.from_numpy(spec.S_db.mean(axis=1).astype(np.float32)).view(1, 1, -1)
        std = torch.from_numpy(np.maximum(spec.S_db.std(axis=1).astype(np.float32), 1e-5)).view(1, 1, -1)

    start = max(0.0, offset_ms)
    timeline_end = spec.times[-1] * 1000.0
    pad_value = float(spec.S_db.min())

    chunks: List[AudioChunk] = []
    while start < timeline_end:
        start_frame = int(round(start / spec.frame_duration_ms))
        mel_slice = spec.S_db[:, start_frame : start_frame + raw_frames_per_chunk : frame_stride]
        valid_frames = mel_slice.shape[1]
        if valid_frames < frames_per_chunk:
            pad = frames_per_chunk - valid_frames
            mel_slice = np.pad(mel_slice, ((0, 0), (0, pad)), constant_values=pad_value)

        audio = torch.from_numpy(mel_slice.T).unsqueeze(0).float()
        if normalize_audio:
            audio = (audio - mean) / std

        mask = torch.arange(frames_per_chunk).unsqueeze(0) >= valid_frames
        chunks.append(
            AudioChunk(
                audio=audio,
                mask=mask,
                start_ms=start,
                tick_duration_ms=tick_duration_ms,
                total_ticks=total_ticks,
                context_ticks=context_ticks,
                target_ticks=target_ticks,
                beat_duration_ms=beat_duration_ms,
            )
        )
        start += sample_hop_ms

    return chunks


def _clamp_coord(value: Optional[float], limit: int) -> Optional[int]:
    if value is None:
        return None
    return int(max(0, min(limit, round(value))))


def _quantize_time(value: float, mode: str, threshold: float = 0.5) -> int:
    if mode == "floor":
        return int(math.floor(value))
    if mode == "ceil":
        return int(math.ceil(value))
    if mode == "custom":
        base = math.floor(value)
        frac = value - base
        if frac >= threshold:
            return int(base + 1)
        return int(base)
    return int(round(value))


def _quantize_slider_length(value: float, mode: str, threshold: float = 0.5) -> float | int:
    if mode == "exact":
        return value
    if mode == "floor":
        return math.floor(value)
    if mode == "ceil":
        return math.ceil(value)
    if mode == "custom":
        base = math.floor(value)
        frac = value - base
        return base + (1 if frac >= threshold else 0)
    if mode == "round":
        return int(round(value))
    return value


def tokens_to_hitobjects(
    events: List[dict],
    chunk: AudioChunk,
    data_cfg: dict,
    suppress_overlaps: bool = True,
) -> List[dict]:
    width = data_cfg["osu_width"]
    height = data_cfg["osu_height"]
    rounding_mode = (data_cfg.get("time_rounding_mode") or "round").lower()
    if rounding_mode not in ("round", "floor", "ceil", "custom"):
        rounding_mode = "round"
    rounding_threshold = float(data_cfg.get("time_rounding_threshold", 0.5) or 0.5)
    rounding_threshold = max(0.0, min(1.0, rounding_threshold))

    length_mode = (data_cfg.get("slider_length_rounding_mode") or "exact").lower()
    if length_mode not in ("exact", "round", "floor", "ceil", "custom"):
        length_mode = "exact"
    length_threshold = float(data_cfg.get("slider_length_rounding_threshold", 0.5) or 0.5)
    length_threshold = max(0.0, min(1.0, length_threshold))

    target_start_ms = chunk.start_ms + chunk.context_ticks * chunk.tick_duration_ms
    cutoff_end = chunk.start_ms + chunk.total_ticks * chunk.tick_duration_ms
    blocked_until = float("-inf")
    results: List[dict] = []

    for event in events:
        time_ms = float(event.get("time", 0.0))
        if time_ms < target_start_ms:
            continue
        if time_ms >= cutoff_end:
            break
        if suppress_overlaps and time_ms < blocked_until:
            continue

        if event.get("type") == "circle":
            x = _clamp_coord(event.get("x"), width)
            y = _clamp_coord(event.get("y"), height)
            if x is None or y is None:
                continue
            quant_time = _quantize_time(time_ms, rounding_mode, rounding_threshold)
            results.append(
                {
                    "type": "circle",
                    "time": quant_time,
                    "x": x,
                    "y": y,
                    "end_time": quant_time,
                }
            )
            if suppress_overlaps:
                blocked_until = max(blocked_until, quant_time + chunk.tick_duration_ms)
            continue

        if event.get("type") != "slider":
            continue

        x = _clamp_coord(event.get("x"), width)
        y = _clamp_coord(event.get("y"), height)
        if x is None or y is None:
            continue
        duration_ticks = max(0, int(event.get("duration_ticks", 0)))
        if duration_ticks <= 0:
            continue
        duration_ms = duration_ticks * chunk.tick_duration_ms
        if time_ms + duration_ms > cutoff_end:
            continue

        slides = max(1, int(event.get("slides", 1)))
        sv_factor = float(event.get("sv") or event.get("sv_factor") or 1.0)
        curve_type = event.get("curve_type") or "L"
        points: List[Tuple[int, int]] = []
        for px, py in event.get("points", []):
            cx = _clamp_coord(px, width)
            cy = _clamp_coord(py, height)
            if cx is not None and cy is not None:
                points.append((cx, cy))
        if not points:
            continue

        beat_length = chunk.beat_duration_ms
        if beat_length <= 0:
            continue
        slider_length = duration_ms * sv_factor * 100.0 / (beat_length * slides)
        slider_length = max(5.0, slider_length)
        slider_length = _quantize_slider_length(float(slider_length), length_mode, length_threshold)

        quant_time = _quantize_time(time_ms, rounding_mode, rounding_threshold)
        results.append(
            {
                "type": "slider",
                "time": quant_time,
                "x": x,
                "y": y,
                "curve_type": curve_type,
                "points": points,
                "slides": slides,
                "length": slider_length,
                "sv_factor": float(sv_factor),
                "end_time": quant_time + int(round(duration_ms)),
            }
        )
        if suppress_overlaps:
            blocked_until = max(blocked_until, quant_time + duration_ms + chunk.tick_duration_ms)

    return results


def extract_context_tokens(
    sequence: torch.Tensor,
    tokenizer: HitObjectTokenizer,
    chunk: AudioChunk,
) -> torch.Tensor:
    context_tokens: List[List[int]] = []
    target_start = chunk.target_ticks
    context_limit = chunk.context_ticks
    for token in sequence.tolist():
        if token[TokenAttr.TYPE] == TokenType.EOS:
            break
        tick = tokenizer.tick_from_id(token[TokenAttr.TICK])
        if tick < target_start:
            continue
        shifted = tick - target_start
        if shifted >= context_limit:
            continue
        token_copy = list(token)
        token_copy[TokenAttr.TICK] = tokenizer.encode_tick(shifted)
        context_tokens.append(token_copy)
    if not context_tokens:
        return torch.zeros(1, 0, TokenAttr.COUNT, dtype=torch.long)
    return torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0)




def merge_events(events: List[dict], tolerance_ms: float, suppress_overlaps: bool = True) -> List[dict]:
    if not events:
        return []
    events.sort(key=lambda e: e["time"])
    if not suppress_overlaps:
        return events
    merged: List[dict] = []
    seen_keys: set = set()
    tol = max(1.0, tolerance_ms)
    active_until = float("-inf")

    for event in events:
        if event["time"] < active_until:
            continue
        bucket = int(round(event["time"] / tol))
        sv_key = int(round(float(event.get("sv_factor") or 0.0) * 1000))
        key = (bucket, event["type"], event.get("x"), event.get("y"), sv_key)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(event)
        active_until = max(active_until, float(event.get("end_time", event["time"])))
    merged.sort(key=lambda e: e["time"])
    return merged


def format_hit_objects(events: Sequence[dict]) -> List[str]:
    lines: List[str] = []
    for event in events:
        if event["type"] == "circle":
            lines.append(f"{event['x']},{event['y']},{event['time']},1,0,0:0:0:0:")
        elif event["type"] == "slider":
            points = "|".join(f"{x}:{y}" for x, y in event["points"])
            curve = f"{event['curve_type']}|{points}"
            length = event.get("length")
            if isinstance(length, float):
                length_str = f"{length:.6f}".rstrip("0").rstrip(".")
            else:
                length_str = str(length)
            lines.append(
                f"{event['x']},{event['y']},{event['time']},2,0,{curve},{event['slides']},{length_str},0:0:0:0:"
            )
    return lines


def replace_section(content: str, section: str, lines: Sequence[str]) -> str:
    pattern = re.compile(rf"\[{section}\]\s*(.*?)(?=\n\[|\Z)", re.DOTALL)
    block = f"[{section}]\n" + "\n".join(lines) + "\n"
    if pattern.search(content):
        return pattern.sub(block, content, count=1)
    return content.rstrip() + "\n" + block


def normalize_slider_multiplier(content: str) -> str:
    return re.sub(r"(SliderMultiplier\s*:\s*)([0-9.]+)", r"\g<1>1", content, count=1)


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


def set_editor_bookmarks(content: str, bookmarks: Sequence[int]) -> str:
    bookmark_line = "Bookmarks: " + ",".join(str(int(b)) for b in bookmarks) if bookmarks else "Bookmarks:"
    pattern = re.compile(r"(\[Editor\]\s*)(.*?)(?=\n\[|\Z)", re.DOTALL)
    match = pattern.search(content)
    if match:
        body = match.group(2).strip().splitlines()
        updated = False
        new_lines: List[str] = []
        for line in body:
            if line.strip().startswith("Bookmarks:"):
                new_lines.append(bookmark_line)
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(bookmark_line)
        new_block = "[Editor]\n" + "\n".join(new_lines) + "\n"
        return content[: match.start()] + new_block + content[match.end() :]
    else:
        return content.rstrip() + "\n[Editor]\n" + bookmark_line + "\n"


def build_default_map(audio_name: str, bpm: float, offset: float, hitobject_lines: Sequence[str]) -> str:
    beat_length = 60000.0 / max(bpm, 1e-3)
    content = [
        "osu file format v14",
        "// AI Osu Beatmap Generator - by JustArtiom",
        "// https://github.com/JustArtiom/osu-bmg",
        "",
        "[General]",
        f"AudioFilename: {audio_name}",
        "AudioLeadIn: 0",
        "PreviewTime: -1",
        "Countdown: 0",
        "SampleSet: Normal",
        "StackLeniency: 0.7",
        "Mode: 0",
        "LetterboxInBreaks: 0",
        "WidescreenStoryboard: 0",
        "",
        "[Metadata]",
        "Title:Generated Track",
        "TitleUnicode:Generated Track",
        "Artist:Unknown",
        "ArtistUnicode:Unknown",
        "Creator:Osu Beatmap Generator - by JustArtiom",
        "Version:Auto",
        "Source:",
        "Tags:generated",
        "BeatmapID:0",
        "BeatmapSetID:0",
        "",
        "[Difficulty]",
        "HPDrainRate:5",
        "CircleSize:4",
        "OverallDifficulty:7",
        "ApproachRate:7",
        "SliderMultiplier:1",
        "SliderTickRate:1",
        "",
        "[TimingPoints]",
        f"{offset:.4f},{beat_length:.4f},4,2,0,50,1,0",
        f"{offset:.4f},-100.0000,4,2,0,50,0,0",
        "",
        "[HitObjects]",
        "\n".join(hitobject_lines),
    ]
    return "\n".join(content) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    audio_path = Path(args.audio).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else audio_path.with_suffix(".generated.osu")

    template_path = Path(args.template).expanduser().resolve() if args.template else None

    bpm = args.bpm
    offset = args.offset
    if template_path and template_path.exists():
        template_bpm, template_offset = get_timing_from_template(template_path)
        if bpm is None:
            bpm = template_bpm
        if offset is None:
            offset = template_offset

    device = resolve_device(args.device)

    payload = torch.load(checkpoint_path, map_location=device)
    tokenizer_meta = payload.get("tokenizer_meta") or {}
    if not tokenizer_meta:
        print("[WARN] Checkpoint missing tokenizer metadata; falling back to config.yaml settings.")
    data_cfg_override = tokenizer_meta.get("data_cfg")
    if data_cfg_override:
        merged_cfg = config["data"].copy()
        merged_cfg.update(data_cfg_override)
        config["data"] = merged_cfg
    data_cfg = config["data"]
    audio_cfg = config["audio"]
    tokenizer = HitObjectTokenizer(data_cfg)
    suppress_overlaps = bool(data_cfg.get("suppress_overlap", True))

    bpm = bpm or data_cfg.get("default_bpm", 120.0)
    offset = offset if offset is not None else 0.0

    spec = load_or_create_mel(audio_path, audio_cfg)
    chunks = build_chunks(spec, data_cfg, audio_cfg, bpm, offset)

    model = ConformerSeq2Seq(config).to(device)
    model.load_state_dict(payload["model_state"])
    saved_attr_sizes = tokenizer_meta.get("attr_sizes")
    if saved_attr_sizes and saved_attr_sizes != model.attr_sizes:
        raise RuntimeError(
            f"Tokenizer attribute sizes in checkpoint ({saved_attr_sizes}) do not match current config ({model.attr_sizes})"
        )
    model.eval()

    all_events: List[dict] = []
    context_tokens = torch.zeros(1, 0, TokenAttr.COUNT, dtype=torch.long)
    for chunk in chunks:
        audio = chunk.audio.to(device)
        mask = chunk.mask.to(device)
        prompt = context_tokens if context_tokens.size(1) > 0 else None
        prompt_tensor = prompt.to(device) if prompt is not None else None
        remaining = tokenizer.seq_len - (prompt.size(1) if prompt is not None else 0)
        remaining = max(0, remaining)
        preds = model.generate(
            audio,
            mask,
            prompt=prompt_tensor,
            max_steps=remaining,
            temperature=args.temperature,
        )
        generated = preds[0].cpu()
        if prompt is not None and prompt.size(1) > 0:
            full_sequence = torch.cat([prompt.cpu()[0], generated], dim=0)
        else:
            full_sequence = generated

        detok_events = tokenizer.detokenize(
            full_sequence.tolist(),
            chunk_start_ms=chunk.start_ms,
            tick_duration_ms=chunk.tick_duration_ms,
            cutoff_tick=chunk.total_ticks,
        )
        chunk_events = tokens_to_hitobjects(
            detok_events,
            chunk,
            data_cfg,
            suppress_overlaps=suppress_overlaps,
        )
        all_events.extend(chunk_events)
        context_tokens = extract_context_tokens(full_sequence, tokenizer, chunk)

    tolerance_ms = data_cfg.get("merge_tolerance_ms", chunks[0].tick_duration_ms if chunks else 10.0)
    discrete_events = merge_events(
        all_events,
        tolerance_ms=tolerance_ms,
        suppress_overlaps=suppress_overlaps,
    )
    hitobject_lines = format_hit_objects(discrete_events)

    beat_length = 60000.0 / max(bpm, 1e-3)

    timing_lines = build_timing_points(offset, beat_length, discrete_events)
    bookmark_mode = (data_cfg.get("bookmark_mode") or "events").lower()
    if bookmark_mode == "chunk":
        bookmark_times = sorted({int(round(chunk.start_ms)) for chunk in chunks})
        max_bookmarks = int(data_cfg.get("max_editor_bookmarks", 256) or 0)
        if max_bookmarks > 0 and len(bookmark_times) > max_bookmarks:
            step = max(1, len(bookmark_times) // max_bookmarks)
            bookmark_times = bookmark_times[::step]
    elif bookmark_mode == "none":
        bookmark_times = []
    else:
        bookmark_times = sorted({int(event["time"]) for event in discrete_events})
        if not bookmark_times and chunks:
            bookmark_times = sorted({int(round(chunk.start_ms)) for chunk in chunks})
        max_bookmarks = int(data_cfg.get("max_editor_bookmarks", 256) or 0)
        if max_bookmarks > 0 and len(bookmark_times) > max_bookmarks:
            step = max(1, len(bookmark_times) // max_bookmarks)
            bookmark_times = bookmark_times[::step]

    if template_path and template_path.exists():
        output_text = template_path.read_text(encoding="utf-8")
    else:
        output_text = build_default_map(audio_path.name, bpm, offset, hitobject_lines)

    output_text = normalize_slider_multiplier(output_text)
    output_text = replace_section(output_text, "TimingPoints", timing_lines)
    output_text = replace_section(output_text, "HitObjects", hitobject_lines)
    output_text = set_editor_bookmarks(output_text, bookmark_times)

    output_path.write_text(output_text, encoding="utf-8")
    print(f"[INFO] Generated {len(hitobject_lines)} hitobjects → {output_path}")


if __name__ == "__main__":
    main()
