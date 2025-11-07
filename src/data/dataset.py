from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.audio import AUDIO_EXTENSIONS, MelSpec, prepare_audio
from src.osu import Beatmap, Circle, Slider
from src.utils.file import collect_files


@dataclass
class SampleDescriptor:
    audio_path: str
    start_frame: int
    frames_per_chunk: int
    token_length: int
    tokens: np.ndarray


class OsuBeatmapDataset(Dataset):
    def __init__(self, config: dict, split: str = "train") -> None:
        self.config = config
        self.split = split
        self.data_cfg = config["data"]
        self.training_cfg = config["training"]
        self.paths_cfg = config["paths"]

        self.base_path = Path(self.paths_cfg["data"])
        self.beats_per_sample = self.data_cfg.get("beats_per_sample", 16)
        self.ticks_per_beat = self.data_cfg.get("ticks_per_beat", 8)
        self.sample_hop_beats = self.data_cfg.get("sample_hop_beats", max(1, self.beats_per_sample // 2))
        self.max_seq_len = config["model"]["decoder"].get("max_seq_len", 64)
        self.normalize_audio = self.data_cfg.get("normalize_audio", True)
        self.skip_empty = self.data_cfg.get("skip_empty_chunks", True)

        self.val_split = self.training_cfg.get("val_split", 0.2)

        self._spec_cache: Dict[str, MelSpec] = {}
        self.samples: List[SampleDescriptor] = []

        osu_files = self._gather_osu_files()
        if not osu_files:
            raise RuntimeError(f"No osu files found under {self.base_path}")

        random.shuffle(osu_files)

        val_count = int(math.ceil(len(osu_files) * self.val_split)) if self.val_split > 0 else 0
        if split == "train":
            selected = osu_files[val_count:] if val_count < len(osu_files) else osu_files
        else:
            if val_count:
                selected = osu_files[: max(1, val_count)]
            else:
                fallback = max(1, len(osu_files) // 5)
                selected = osu_files[:fallback]

        if not selected:
            selected = osu_files

        for osu_path in selected:
            self.samples.extend(self._build_samples_for_map(osu_path))

        if not self.samples:
            raise RuntimeError(f"No training samples could be built for split '{split}'")

    def _gather_osu_files(self) -> List[Path]:
        rel_paths = collect_files(
            str(self.base_path),
            include_patterns=["*.osu"],
            exclude_patterns=self.paths_cfg.get("exclude", []),
        )
        return [self.base_path / rel_path for rel_path in rel_paths if rel_path.lower().endswith(".osu")]

    def _build_samples_for_map(self, osu_path: Path) -> List[SampleDescriptor]:
        beatmap = Beatmap(file_path=str(osu_path))
        audio_path = self._resolve_audio_path(osu_path, beatmap)
        spec = self._load_mel_spec(audio_path)

        bpm, offset_ms = self._get_primary_bpm_and_offset(beatmap)
        if bpm <= 0:
            bpm = self.data_cfg.get("default_bpm", 120.0)

        events = self._collect_events(beatmap)
        events.sort(key=lambda item: item[0])

        ticks_per_sample = self.beats_per_sample * self.ticks_per_beat
        beat_duration_ms = 60000.0 / max(bpm, 1e-3)
        chunk_duration_ms = self.beats_per_sample * beat_duration_ms
        sample_hop_ms = self.sample_hop_beats * beat_duration_ms
        tick_duration_ms = beat_duration_ms / self.ticks_per_beat
        frames_per_chunk = max(1, int(round(chunk_duration_ms / spec.frame_duration_ms)))

        timeline_end_ms = spec.times[-1] * 1000.0
        current_start = max(0.0, offset_ms)

        descriptors: List[SampleDescriptor] = []
        while current_start < timeline_end_ms:
            chunk_end = current_start + chunk_duration_ms
            chunk_events = self._filter_events(events, current_start, chunk_end)

            if not chunk_events and self.skip_empty:
                current_start += sample_hop_ms
                continue

            tokens, token_length = self._encode_events(
                chunk_events,
                current_start,
                ticks_per_sample,
                tick_duration_ms,
            )

            if token_length == 0:
                current_start += sample_hop_ms
                continue

            start_frame = int(round(current_start / spec.frame_duration_ms))
            descriptors.append(
                SampleDescriptor(
                    audio_path=str(audio_path),
                    start_frame=max(0, start_frame),
                    frames_per_chunk=frames_per_chunk,
                    token_length=token_length,
                    tokens=tokens,
                )
            )
            current_start += sample_hop_ms

        if not descriptors and events:
            tokens, token_length = self._encode_events(events, max(0.0, offset_ms), ticks_per_sample, tick_duration_ms)
            descriptors.append(
                SampleDescriptor(
                    audio_path=str(audio_path),
                    start_frame=0,
                    frames_per_chunk=max(1, int(round(chunk_duration_ms / spec.frame_duration_ms))),
                    token_length=token_length,
                    tokens=tokens,
                )
            )

        return descriptors

    def _resolve_audio_path(self, osu_path: Path, beatmap: Beatmap) -> Path:
        if hasattr(beatmap, "general") and beatmap.general.audio_filename:
            candidate = osu_path.parent / beatmap.general.audio_filename
            if candidate.exists():
                return candidate

        for pattern in AUDIO_EXTENSIONS:
            for candidate in osu_path.parent.glob(pattern):
                if candidate.exists():
                    return candidate

        raise FileNotFoundError(f"Audio file not found for beatmap {osu_path}")

    def _load_mel_spec(self, audio_path: Path) -> MelSpec:
        audio_key = str(audio_path)
        if audio_key in self._spec_cache:
            return self._spec_cache[audio_key]

        npz_path = Path(str(audio_path) + ".mel.npz")
        if not npz_path.exists():
            prepare_audio(
                str(audio_path),
                self.data_cfg["sample_rate"],
                self.data_cfg["hop_ms"],
                self.data_cfg["win_ms"],
                self.data_cfg["n_mels"],
                self.data_cfg["n_fft"],
                force=False,
            )
        spec = MelSpec.load_npz(str(npz_path))
        self._spec_cache[audio_key] = spec
        return spec

    def _get_primary_bpm_and_offset(self, beatmap: Beatmap) -> Tuple[float, float]:
        for tp in getattr(beatmap, "timing_points", []):
            if tp.uninherited == 1:
                return tp.get_bpm(), tp.time
        return self.data_cfg.get("default_bpm", 120.0), 0.0

    def _collect_events(self, beatmap: Beatmap) -> List[Tuple[float, float, float]]:
        events: List[Tuple[float, float, float]] = []
        for ho in getattr(beatmap, "hit_objects", []):
            if isinstance(ho, Circle):
                events.append((float(ho.time), float(ho.x), float(ho.y)))
            elif isinstance(ho, Slider):
                events.extend(self._slider_to_events(ho))
        return events

    def _slider_to_events(self, slider: Slider) -> List[Tuple[float, float, float]]:
        points: List[Tuple[float, float]] = []
        if slider.object_params and slider.object_params.curves:
            for curve in slider.object_params.curves:
                points.extend(curve.curve_points)

        end_point = points[-1] if points else (slider.x, slider.y)
        if slider.object_params and slider.object_params.slides % 2 == 0:
            end_point = (slider.x, slider.y)

        duration = slider.object_params.duration if slider.object_params else 0.0

        events = [
            (float(slider.time), float(slider.x), float(slider.y)),
        ]
        if duration > 0:
            events.append((float(slider.time + duration), float(end_point[0]), float(end_point[1])))
        return events

    def _filter_events(
        self,
        events: Sequence[Tuple[float, float, float]],
        start_ms: float,
        end_ms: float,
    ) -> List[Tuple[float, float, float]]:
        return [event for event in events if start_ms <= event[0] < end_ms]

    def _encode_events(
        self,
        events: Sequence[Tuple[float, float, float]],
        chunk_start_ms: float,
        ticks_per_sample: int,
        tick_duration_ms: float,
    ) -> Tuple[np.ndarray, int]:
        encoded: List[Tuple[float, float, float, float]] = []
        for event in events:
            rel_ms = max(0.0, event[0] - chunk_start_ms)
            tick_idx = min(ticks_per_sample - 1, rel_ms / max(tick_duration_ms, 1e-3))
            tick_norm = float(tick_idx) / ticks_per_sample
            x_norm = np.clip(event[1] / self.data_cfg["osu_width"], 0.0, 1.0)
            y_norm = np.clip(event[2] / self.data_cfg["osu_height"], 0.0, 1.0)
            encoded.append((tick_norm, x_norm, y_norm, 0.0))

        if len(encoded) >= self.max_seq_len:
            encoded = encoded[: self.max_seq_len - 1]

        encoded.append((0.0, 0.0, 0.0, 1.0))

        token_length = min(len(encoded), self.max_seq_len)
        tokens = np.zeros((self.max_seq_len, 4), dtype=np.float32)
        tokens[:token_length] = np.asarray(encoded[:token_length], dtype=np.float32)
        if token_length < self.max_seq_len:
            tokens[token_length:, 3] = 1.0

        return tokens, token_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        descriptor = self.samples[idx]
        spec = self._spec_cache[descriptor.audio_path]

        start = descriptor.start_frame
        end = start + descriptor.frames_per_chunk
        mel_slice = spec.S_db[:, start:end]
        valid_frames = mel_slice.shape[1]

        if valid_frames < descriptor.frames_per_chunk:
            pad_width = descriptor.frames_per_chunk - valid_frames
            pad_value = float(spec.S_db.min())
            mel_slice = np.pad(mel_slice, ((0, 0), (0, pad_width)), constant_values=pad_value)

        mel = torch.from_numpy(mel_slice.T).float()
        audio_length = valid_frames

        if self.normalize_audio:
            mean = mel.mean(dim=0, keepdim=True)
            std = mel.std(dim=0, keepdim=True).clamp_min(1e-5)
            mel = (mel - mean) / std

        tokens = torch.from_numpy(descriptor.tokens).float()

        return {
            "audio": mel,
            "audio_length": audio_length,
            "tokens": tokens,
            "token_length": descriptor.token_length,
        }


def osu_collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    audio_dim = batch[0]["audio"].shape[-1]
    token_dim = batch[0]["tokens"].shape[-1]

    max_audio_len = max(item["audio"].shape[0] for item in batch)
    max_token_len = max(item["tokens"].shape[0] for item in batch)

    audio_dtype = batch[0]["audio"].dtype
    token_dtype = batch[0]["tokens"].dtype

    audio_batch = torch.zeros(batch_size, max_audio_len, audio_dim, dtype=audio_dtype)
    token_batch = torch.zeros(batch_size, max_token_len, token_dim, dtype=token_dtype)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    token_lengths = torch.zeros(batch_size, dtype=torch.long)

    for idx, item in enumerate(batch):
        audio = item["audio"]
        tokens = item["tokens"]

        audio_batch[idx, : audio.shape[0]] = audio
        token_batch[idx, : tokens.shape[0]] = tokens
        audio_lengths[idx] = audio.shape[0]
        token_lengths[idx] = item["token_length"]

    audio_mask = torch.arange(max_audio_len).unsqueeze(0) >= audio_lengths.unsqueeze(1)
    token_mask = torch.arange(max_token_len).unsqueeze(0) >= token_lengths.unsqueeze(1)

    return {
        "audio": audio_batch,
        "audio_mask": audio_mask,
        "audio_lengths": audio_lengths,
        "tokens": token_batch,
        "token_mask": token_mask,
        "token_lengths": token_lengths,
    }
