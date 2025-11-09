from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.audio import AUDIO_EXTENSIONS, MelSpec, prepare_audio
from src.osu import Beatmap, Circle, Slider
from src.utils.file import collect_files

try:  # pragma: no cover - tqdm optional at runtime, but usually installed
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

_SPLIT_CACHE: Dict[Tuple[str, float, Optional[int], Tuple[str, ...]], Tuple[List[Path], List[Path]]] = {}


@dataclass
class SampleDescriptor:
    audio_path: str
    start_frame: int
    frames_per_chunk: int
    token_length: int
    tokens: np.ndarray
    frame_stride: int = 1
    raw_frames: int = 0


def collect_osu_file_paths(paths_cfg: dict) -> List[Path]:
    base_path = Path(paths_cfg["data"])
    rel_paths = collect_files(
        str(base_path),
        include_patterns=["*.osu"],
        exclude_patterns=paths_cfg.get("exclude", []),
    )
    return [base_path / rel_path for rel_path in rel_paths if rel_path.lower().endswith(".osu")]


def split_train_val_files(
    config: dict,
    osu_files: Optional[Sequence[Path]] = None,
) -> Tuple[List[Path], List[Path]]:
    paths_cfg = config["paths"]
    training_cfg = config["training"]
    cache_key = None
    if osu_files is None:
        base_path = str(Path(paths_cfg["data"]).resolve())
        val_split = training_cfg.get("val_split", 0.2)
        dataset_seed = training_cfg.get("dataset_seed")
        exclude_patterns = tuple(sorted(paths_cfg.get("exclude", [])))
        cache_key = (base_path, float(val_split), dataset_seed, exclude_patterns)
        if cache_key in _SPLIT_CACHE:
            cached_train, cached_val = _SPLIT_CACHE[cache_key]
            return list(cached_train), list(cached_val)

    files = list(osu_files) if osu_files is not None else collect_osu_file_paths(paths_cfg)
    if not files:
        raise RuntimeError(f"No osu files found under {paths_cfg['data']}")

    dataset_seed = training_cfg.get("dataset_seed")
    rng = random.Random(dataset_seed)
    rng.shuffle(files)

    val_split = training_cfg.get("val_split", 0.2)
    val_count = int(math.ceil(len(files) * val_split)) if val_split > 0 else 0
    if val_count > 0:
        val_size = max(1, val_count)
        val_files = files[:val_size]
        train_files = files[val_size:] or files
    else:
        fallback = max(1, len(files) // 5)
        val_files = files[:fallback]
        train_files = files[fallback:] or files

    result = (train_files, val_files)
    if cache_key is not None:
        _SPLIT_CACHE[cache_key] = (train_files.copy(), val_files.copy())
    return train_files, val_files


class OsuBeatmapDataset(Dataset):
    def __init__(self, config: dict, split: str = "train", osu_files: Optional[Sequence[Path]] = None) -> None:
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
        self.min_star_rating = self._parse_star_rating(self.data_cfg.get("min_star_rating"), "min_star_rating")
        self.max_star_rating = self._parse_star_rating(self.data_cfg.get("max_star_rating"), "max_star_rating")
        self.min_bpm = self._parse_bpm(self.data_cfg.get("min_bpm"), "min_bpm")
        self.max_bpm = self._parse_bpm(self.data_cfg.get("max_bpm"), "max_bpm")
        if (
            self.min_star_rating is not None
            and self.max_star_rating is not None
            and self.min_star_rating > self.max_star_rating
        ):
            raise ValueError("data.min_star_rating cannot be greater than data.max_star_rating")

        self._spec_cache: Dict[str, MelSpec] = {}
        self._spec_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.samples: List[SampleDescriptor] = []

        if osu_files is None:
            train_files, val_files = split_train_val_files(config)
            osu_files = train_files if split == "train" else val_files

        selected = list(osu_files)
        if not selected:
            raise RuntimeError(f"No osu files available for split '{split}'")

        progress_bar = None
        iterator: Sequence[Path]
        if tqdm is not None:
            progress_bar = tqdm(selected, desc=f"Loading {self.split} beatmaps", unit="map")
            iterator = progress_bar
        else:
            iterator = selected

        for osu_path in iterator:
            beatmap = self._load_standard_beatmap(osu_path)
            if beatmap is None:
                continue
            self.samples.extend(self._build_samples_for_map(osu_path, beatmap))
            if progress_bar is not None:
                progress_bar.set_postfix(samples=len(self.samples))

        if progress_bar is not None:
            progress_bar.close()

        if not self.samples:
            raise RuntimeError(f"No training samples could be built for split '{split}'")

    def _compute_frame_stride(self, frames: int) -> Tuple[int, int]:
        frames = max(1, frames)
        max_frames = self.data_cfg.get("max_frames_per_sample")
        if not max_frames or max_frames <= 0 or frames <= max_frames:
            return 1, frames
        stride = int(math.ceil(frames / max_frames))
        adjusted = int(math.ceil(frames / stride))
        return max(1, stride), max(1, adjusted)

    def _compute_spec_stats(self, spec: MelSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        data = spec.S_db
        mean = torch.from_numpy(data.mean(axis=1).astype(np.float32))
        std = torch.from_numpy(np.maximum(data.std(axis=1).astype(np.float32), 1e-5))
        return mean, std

    def _get_spec_stats(self, audio_key: str, spec: MelSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio_key not in self._spec_stats:
            self._spec_stats[audio_key] = self._compute_spec_stats(spec)
        return self._spec_stats[audio_key]

    def _load_standard_beatmap(self, osu_path: Path) -> Beatmap | None:
        try:
            beatmap = Beatmap(file_path=str(osu_path))
        except:  # pragma: no cover - defensive guard
            return None

        mode = 0
        general = getattr(beatmap, "general", None)
        if general is not None:
            try:
                mode = int(getattr(general, "mode", 0))
            except (TypeError, ValueError):
                mode = 0

        if mode != 0:
            return None

        return beatmap

    def _build_samples_for_map(self, osu_path: Path, beatmap: Beatmap) -> List[SampleDescriptor]:
        if not self._star_rating_in_range(beatmap, osu_path):
            return []
        if not self._bpm_in_range(beatmap, osu_path):
            return []
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
        raw_frames_per_chunk = max(1, int(round(chunk_duration_ms / spec.frame_duration_ms)))
        frame_stride, frames_per_chunk = self._compute_frame_stride(raw_frames_per_chunk)
        if frame_stride > 1:
            print(
                f"[INFO] Downsampling audio window for {osu_path.name}: "
                f"{raw_frames_per_chunk} frames -> {frames_per_chunk} (stride {frame_stride})"
            )

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
                    frame_stride=frame_stride,
                    raw_frames=raw_frames_per_chunk,
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
                    frames_per_chunk=frames_per_chunk,
                    frame_stride=frame_stride,
                    raw_frames=raw_frames_per_chunk,
                    token_length=token_length,
                    tokens=tokens,
                )
            )

        return descriptors

    @staticmethod
    def _parse_star_rating(value, key: str) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"data.{key} must be a number or null, got {value!r}") from exc

    @staticmethod
    def _parse_bpm(value, key: str) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"data.{key} must be a number or null, got {value!r}") from exc

    def _star_rating_in_range(self, beatmap: Beatmap, osu_path: Path) -> bool:
        if self.min_star_rating is None and self.max_star_rating is None:
            return True
        try:
            difficulty = beatmap.get_difficulty()
            star_rating = float(getattr(difficulty, "star_rating", 0.0))
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"[WARN] Skipping {osu_path} due to star difficulty calculation failure: {exc}")
            return False

        if self.min_star_rating is not None and star_rating < self.min_star_rating:
            return False
        if self.max_star_rating is not None and star_rating > self.max_star_rating:
            return False
        return True

    def _bpm_in_range(self, beatmap: Beatmap, osu_path: Path) -> bool:
        if self.min_bpm is None and self.max_bpm is None:
            return True
        bpm = None
        for tp in getattr(beatmap, "timing_points", []):
            if tp.uninherited == 1:
                bpm = tp.get_bpm()
                break
        if bpm is None:
            print(f"[WARN] Skipping {osu_path}: could not determine BPM.")
            return False
        if self.min_bpm is not None and bpm < self.min_bpm:
            return False
        if self.max_bpm is not None and bpm > self.max_bpm:
            return False
        return True

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
        if self.normalize_audio:
            self._spec_stats[audio_key] = self._compute_spec_stats(spec)
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
        raw_frames = descriptor.raw_frames or (descriptor.frames_per_chunk * descriptor.frame_stride)
        end = start + raw_frames
        stride = max(1, descriptor.frame_stride)
        mel_slice = spec.S_db[:, start:end:stride]
        valid_frames = mel_slice.shape[1]

        if valid_frames < descriptor.frames_per_chunk:
            pad_width = descriptor.frames_per_chunk - valid_frames
            pad_value = float(spec.S_db.min())
            mel_slice = np.pad(mel_slice, ((0, 0), (0, pad_width)), constant_values=pad_value)

        mel = torch.from_numpy(mel_slice.T).float()
        audio_length = valid_frames

        if self.normalize_audio:
            mean, std = self._get_spec_stats(descriptor.audio_path, spec)
            mel = (mel - mean.unsqueeze(0)) / std.unsqueeze(0)

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
