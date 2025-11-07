from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

from src.audio import MelSpec, prepare_audio
from src.models import ConformerSeq2Seq
from src.osu import Beatmap
from src.utils.config import load_config


@dataclass
class AudioChunk:
    audio: torch.Tensor
    mask: torch.Tensor
    start_ms: float
    tick_duration_ms: float
    ticks_per_sample: int


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
    parser.add_argument("--eos-threshold", type=float, default=0.6, help="Probability threshold for stopping decoding.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        name = "cpu"
    return torch.device(name)


def load_or_create_mel(audio_path: Path, data_cfg: dict) -> MelSpec:
    npz_path = Path(str(audio_path) + ".mel.npz")
    if not npz_path.exists():
        prepare_audio(
            str(audio_path),
            data_cfg["sample_rate"],
            data_cfg["hop_ms"],
            data_cfg["win_ms"],
            data_cfg["n_mels"],
            data_cfg["n_fft"],
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


def build_chunks(spec: MelSpec, data_cfg: dict, bpm: float, offset_ms: float) -> List[AudioChunk]:
    beats_per_sample = data_cfg.get("beats_per_sample", 16)
    ticks_per_beat = data_cfg.get("ticks_per_beat", 8)
    sample_hop_beats = data_cfg.get("sample_hop_beats", max(1, beats_per_sample // 2))
    ticks_per_sample = beats_per_sample * ticks_per_beat

    beat_duration_ms = 60000.0 / max(bpm, 1e-3)
    chunk_duration_ms = beats_per_sample * beat_duration_ms
    sample_hop_ms = sample_hop_beats * beat_duration_ms
    tick_duration_ms = beat_duration_ms / ticks_per_beat
    frames_per_chunk = max(1, int(round(chunk_duration_ms / spec.frame_duration_ms)))

    start = max(0.0, offset_ms)
    timeline_end = spec.times[-1] * 1000.0
    pad_value = float(spec.S_db.min())

    chunks: List[AudioChunk] = []
    while start < timeline_end:
        start_frame = int(round(start / spec.frame_duration_ms))
        mel_slice = spec.S_db[:, start_frame : start_frame + frames_per_chunk]
        valid_frames = mel_slice.shape[1]
        if valid_frames < frames_per_chunk:
            pad = frames_per_chunk - valid_frames
            mel_slice = np.pad(mel_slice, ((0, 0), (0, pad)), constant_values=pad_value)

        audio = torch.from_numpy(mel_slice.T).unsqueeze(0).float()
        if data_cfg.get("normalize_audio", True):
            mean = audio.mean(dim=1, keepdim=True)
            std = audio.std(dim=1, keepdim=True).clamp_min(1e-5)
            audio = (audio - mean) / std

        mask = torch.arange(frames_per_chunk).unsqueeze(0) >= valid_frames
        chunks.append(AudioChunk(audio=audio, mask=mask, start_ms=start, tick_duration_ms=tick_duration_ms, ticks_per_sample=ticks_per_sample))
        start += sample_hop_ms

    return chunks


def tokens_to_events(
    tokens: torch.Tensor,
    chunk: AudioChunk,
    data_cfg: dict,
    eos_threshold: float,
) -> List[Tuple[float, float, float]]:
    events: List[Tuple[float, float, float]] = []
    for token in tokens:
        tick_norm, x_norm, y_norm, eos_prob = token.tolist()
        if eos_prob > eos_threshold:
            break
        tick_idx = tick_norm * chunk.ticks_per_sample
        time_ms = chunk.start_ms + tick_idx * chunk.tick_duration_ms
        x = np.clip(x_norm, 0.0, 1.0) * data_cfg["osu_width"]
        y = np.clip(y_norm, 0.0, 1.0) * data_cfg["osu_height"]
        events.append((time_ms, x, y))
    return events


def merge_events(events: List[Tuple[float, float, float]], tolerance_ms: float, data_cfg: dict) -> List[Tuple[int, int, int]]:
    if not events:
        return []
    events.sort(key=lambda e: e[0])
    merged: List[Tuple[float, float, float]] = [events[0]]
    for time_ms, x, y in events[1:]:
        last_time, last_x, last_y = merged[-1]
        if time_ms - last_time <= tolerance_ms:
            merged[-1] = ((last_time + time_ms) / 2.0, (last_x + x) / 2.0, (last_y + y) / 2.0)
        else:
            merged.append((time_ms, x, y))
    discrete = [
        (
            int(round(t)),
            int(round(np.clip(x, 0.0, data_cfg["osu_width"]))),
            int(round(np.clip(y, 0.0, data_cfg["osu_height"]))),
        )
        for t, x, y in merged
    ]
    return discrete


def format_hit_objects(events: Sequence[Tuple[int, int, int]]) -> List[str]:
    lines = []
    for time_ms, x, y in events:
        lines.append(f"{x},{y},{time_ms},1,0,0:0:0:0:")
    return lines


def inject_hitobjects(template: str, hitobject_lines: Sequence[str]) -> str:
    pattern = re.compile(r"(\[HitObjects\]\s*)(.*?)(?=\n\[|\Z)", re.DOTALL)
    replacement = "\\1" + "\n".join(hitobject_lines) + "\n"
    updated, count = pattern.subn(replacement, template)
    if count == 0:
        updated = template.rstrip() + "\n[HitObjects]\n" + "\n".join(hitobject_lines) + "\n"
    return updated


def build_default_map(audio_name: str, bpm: float, offset: float, hitobject_lines: Sequence[str]) -> str:
    beat_length = 60000.0 / max(bpm, 1e-3)
    content = [
        "osu file format v14",
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
        "Creator:Codex",
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
        "SliderMultiplier:1.4",
        "SliderTickRate:1",
        "",
        "[TimingPoints]",
        f"{offset:.4f},{beat_length:.4f},4,2,0,50,1,0",
        "",
        "[HitObjects]",
        "\n".join(hitobject_lines),
    ]
    return "\n".join(content) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
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

    bpm = bpm or data_cfg.get("default_bpm", 120.0)
    offset = offset if offset is not None else 0.0

    device = resolve_device(args.device)

    spec = load_or_create_mel(audio_path, data_cfg)
    chunks = build_chunks(spec, data_cfg, bpm, offset)

    model = ConformerSeq2Seq(config).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    all_events: List[Tuple[float, float, float]] = []
    for chunk in chunks:
        audio = chunk.audio.to(device)
        mask = chunk.mask.to(device)
        preds = model.generate(audio, mask, eos_threshold=args.eos_threshold)
        chunk_events = tokens_to_events(preds[0].cpu(), chunk, data_cfg, args.eos_threshold)
        all_events.extend(chunk_events)

    tolerance_ms = data_cfg.get("merge_tolerance_ms", chunks[0].tick_duration_ms if chunks else 10.0)
    discrete_events = merge_events(all_events, tolerance_ms=tolerance_ms, data_cfg=data_cfg)
    hitobject_lines = format_hit_objects(discrete_events)

    if template_path and template_path.exists():
        template_text = template_path.read_text(encoding="utf-8")
        output_text = inject_hitobjects(template_text, hitobject_lines)
    else:
        output_text = build_default_map(audio_path.name, bpm, offset, hitobject_lines)

    output_path.write_text(output_text, encoding="utf-8")
    print(f"[INFO] Generated {len(hitobject_lines)} hitobjects → {output_path}")


if __name__ == "__main__":
    main()
