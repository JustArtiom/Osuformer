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
from src.data.tokenizer import HitObjectTokenizer, TokenAttr, TokenType
from src.osu import Beatmap, Circle, Slider
from src.utils.file import collect_files
import json

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
    loss_mask: np.ndarray
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
    def __init__(
        self,
        config: dict,
        split: str = "train",
        osu_files: Optional[Sequence[Path]] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.split = split
        self.data_cfg = config["data"]
        self.audio_cfg = config["audio"]
        self.filters_cfg = config.get("filters", {})
        self.training_cfg = config["training"]
        self.paths_cfg = config["paths"]

        self.base_path = Path(self.paths_cfg["data"])
        self.ticks_per_beat = int(self.data_cfg.get("ticks_per_beat", 4))
        self.context_beats = int(self.data_cfg.get("context_beats", 8))
        self.target_beats = int(self.data_cfg.get("target_beats", 16))
        self.total_beats = self.context_beats + self.target_beats
        self.sample_hop_beats = int(self.data_cfg.get("sample_hop_beats", self.target_beats))
        self.sample_hop_beats = max(1, self.sample_hop_beats)
        self.seq_len = self.total_beats * self.ticks_per_beat
        self.context_ticks = self.context_beats * self.ticks_per_beat
        self.target_ticks = self.target_beats * self.ticks_per_beat
        self.use_spec_cache = bool(self.data_cfg.get("use_spec_cache", True))
        self.normalize_audio = bool(self.audio_cfg.get("normalize_audio", True))
        self.skip_empty = bool(self.audio_cfg.get("skip_empty_chunks", False))
        self.tick_tolerance_ms = float(self.data_cfg.get("tick_tolerance_ms", 3.0))
        self.require_constant_bpm = bool(self._filter_value("require_constant_bpm", False))
        self.require_tick_alignment = bool(self._filter_value("require_tick_alignment", False))
        self.map_types = {mt.lower() for mt in (self._filter_value("map_types", []) or [])}
        type_thresholds = self._filter_value("map_type_thresholds", {}) or {}
        self.stream_interval_ms = float(type_thresholds.get("stream_interval_ms", 110.0))
        self.stream_ratio = float(type_thresholds.get("stream_ratio", 0.35))
        self.stream_gap_distance = float(type_thresholds.get("stream_gap_distance", 80.0))
        self.aim_jump_distance = float(type_thresholds.get("aim_jump_distance", 120.0))
        self.aim_ratio = float(type_thresholds.get("aim_ratio", 0.3))
        self.tokenizer = HitObjectTokenizer(self.data_cfg)
        self.attr_sizes = self.tokenizer.attribute_sizes
        self.cache_path = cache_path
        self.min_star_rating = self._parse_star_rating(self._filter_value("min_star_rating"), "min_star_rating")
        self.max_star_rating = self._parse_star_rating(self._filter_value("max_star_rating"), "max_star_rating")
        self.min_bpm = self._parse_bpm(self._filter_value("min_bpm"), "min_bpm")
        self.max_bpm = self._parse_bpm(self._filter_value("max_bpm"), "max_bpm")
        self.min_rating = self._parse_rating(self._filter_value("min_rating"), "min_rating")
        self.max_rating = self._parse_rating(self._filter_value("max_rating"), "max_rating")
        self.rating_filters_active = self.min_rating is not None or self.max_rating is not None
        if (
            self.min_star_rating is not None
            and self.max_star_rating is not None
            and self.min_star_rating > self.max_star_rating
        ):
            raise ValueError("data.min_star_rating cannot be greater than data.max_star_rating")

        self._spec_cache: Dict[str, MelSpec] = {}
        self._spec_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.samples: List[SampleDescriptor] = []
        self._stats_cache: Dict[str, Optional[float]] = {}

        cache_loaded = False
        if self.cache_path and Path(self.cache_path).exists():
            try:
                self._load_cache(Path(self.cache_path))
                cache_loaded = True
                print(f"[INFO] Loaded {len(self.samples)} {self.split} samples from cache {self.cache_path}")
            except Exception as exc:
                print(f"[WARN] Failed to load cache '{self.cache_path}': {exc}. Rebuilding...")
                self.samples = []

        if cache_loaded:
            return

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

        if self.cache_path:
            try:
                self._save_cache(Path(self.cache_path))
                print(f"[INFO] Saved {len(self.samples)} {self.split} samples to cache {self.cache_path}")
            except Exception as exc:
                print(f"[WARN] Failed to save cache '{self.cache_path}': {exc}")

    def _compute_frame_stride(self, frames: int) -> Tuple[int, int]:
        frames = max(1, frames)
        max_frames = self.audio_cfg.get("max_frames_per_sample")
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

    def _filter_value(self, key: str, default=None):
        if key in self.filters_cfg:
            return self.filters_cfg.get(key, default)
        return self.data_cfg.get(key, default)

    def _get_spec_stats(self, audio_key: str, spec: MelSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_spec_cache:
            return self._compute_spec_stats(spec)
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
        if self.rating_filters_active:
            rating_value = self._get_user_rating(osu_path)
            if rating_value is None:
                print(f"[WARN] Skipping {osu_path}: missing stats for rating filter.")
                return []
            if not self._rating_in_range(rating_value):
                return []
        if self.require_constant_bpm and not self._has_constant_bpm(beatmap):
            return []
        events = self._collect_events(beatmap)
        events.sort(key=lambda item: item[0])
        if not events:
            return []
        if not self._map_types_match(events):
            return []

        audio_path = self._resolve_audio_path(osu_path, beatmap)
        spec = self._load_mel_spec(audio_path)

        bpm, offset_ms = self._get_primary_bpm_and_offset(beatmap)
        if bpm <= 0:
            bpm = self.data_cfg.get("default_bpm", 120.0)
        if self.require_tick_alignment and not self._aligned_to_tick_grid(beatmap, bpm, offset_ms):
            return []

        ticks_per_sample = self.seq_len
        beat_duration_ms = 60000.0 / max(bpm, 1e-3)
        chunk_duration_ms = self.total_beats * beat_duration_ms
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
            chunk_hit_objects = self._filter_hit_objects(beatmap, current_start, chunk_end)

            if not chunk_hit_objects and self.skip_empty:
                current_start += sample_hop_ms
                continue

            encoded = self._encode_chunk_tokens(
                chunk_hit_objects,
                current_start,
                ticks_per_sample,
                tick_duration_ms,
                beatmap,
            )
            if encoded is None:
                current_start += sample_hop_ms
                continue
            tokens, token_length, loss_mask = encoded

            start_frame = int(round(current_start / spec.frame_duration_ms))
            descriptors.append(
                SampleDescriptor(
                    audio_path=str(audio_path),
                    start_frame=max(0, start_frame),
                    frames_per_chunk=frames_per_chunk,
                    frame_stride=frame_stride,
                    raw_frames=raw_frames_per_chunk,
                    token_length=token_length,
                    tokens=np.asarray(tokens, dtype=np.int64),
                    loss_mask=np.asarray(loss_mask, dtype=np.float32),
                )
            )
            current_start += sample_hop_ms

        if not descriptors and events:
            encoded = self._encode_chunk_tokens(
                getattr(beatmap, "hit_objects", []),
                max(0.0, offset_ms),
                ticks_per_sample,
                tick_duration_ms,
                beatmap,
            )
            if encoded is None:
                return []
            tokens, token_length, loss_mask = encoded
            descriptors.append(
                SampleDescriptor(
                    audio_path=str(audio_path),
                    start_frame=0,
                    frames_per_chunk=frames_per_chunk,
                    frame_stride=frame_stride,
                    raw_frames=raw_frames_per_chunk,
                    token_length=token_length,
                    tokens=np.asarray(tokens, dtype=np.int64),
                    loss_mask=np.asarray(loss_mask, dtype=np.float32),
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

    @staticmethod
    def _parse_rating(value, key: str) -> float | None:
        if value is None:
            return None
        try:
            val = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"filters.{key} must be a number or null, got {value!r}") from exc
        if not (0.0 <= val <= 10.0):
            raise ValueError(f"filters.{key} must be between 0 and 10, got {val}")
        return val

    def _get_user_rating(self, osu_path: Path) -> Optional[float]:
        stats_path = osu_path.with_suffix(".stats.json")
        key = str(stats_path)
        if key in self._stats_cache:
            return self._stats_cache[key]
        if not stats_path.exists():
            self._stats_cache[key] = None
            return None
        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to read stats file '{stats_path}': {exc}")
            self._stats_cache[key] = None
            return None
        beatmap_data = payload.get("data") if isinstance(payload, dict) else None
        rating_counts = None
        if isinstance(beatmap_data, dict):
            beatmapset = beatmap_data.get("beatmapset")
            if isinstance(beatmapset, dict):
                rating_counts = beatmapset.get("ratings")
        rating_value = self._compute_rating_from_counts(rating_counts)
        self._stats_cache[key] = rating_value
        return rating_value

    @staticmethod
    def _compute_rating_from_counts(ratings: Optional[Sequence[int]]) -> Optional[float]:
        if not ratings:
            return None
        total = sum(ratings)
        if total <= 0:
            return None
        weighted = sum(idx * count for idx, count in enumerate(ratings))
        return float(weighted) / float(total)

    def _rating_in_range(self, rating: float) -> bool:
        if rating is None:
            return False
        if self.min_rating is not None and rating < self.min_rating:
            return False
        if self.max_rating is not None and rating > self.max_rating:
            return False
        return True

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

    def _has_constant_bpm(self, beatmap: Beatmap) -> bool:
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

    def _effective_slider_sv(self, beatmap: Beatmap, time_ms: float) -> float:
        difficulty = getattr(beatmap, "difficulty", None)
        base_sv = float(getattr(difficulty, "slider_multiplier", 1.0))
        inherited_tp = beatmap.get_previous_timing_point(time_ms, filter=lambda tp: tp.uninherited == 0)
        if inherited_tp is not None:
            sv_multiplier = inherited_tp.get_slider_velocity_multiplier()
        else:
            sv_multiplier = 1.0
        return max(1e-3, base_sv * sv_multiplier)

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

    def _aligned_to_tick_grid(self, beatmap: Beatmap, bpm: float, offset_ms: float) -> bool:
        tick_duration_ms = 60000.0 / max(bpm, 1e-3) / max(self.ticks_per_beat, 1)
        if tick_duration_ms <= 0:
            return False
        tolerance = self.tick_tolerance_ms
        for ho in getattr(beatmap, "hit_objects", []):
            time_ms = float(getattr(ho, "time", 0.0))
            rel = time_ms - offset_ms
            nearest_ticks = round(rel / tick_duration_ms)
            aligned_time = nearest_ticks * tick_duration_ms
            if abs(rel - aligned_time) > tolerance:
                return False
        return True

    def _load_mel_spec(self, audio_path: Path | str) -> MelSpec:
        apath = Path(audio_path)
        audio_key = str(apath)
        if audio_key in self._spec_cache:
            return self._spec_cache[audio_key]

        npz_path = Path(str(apath) + ".mel.npz")
        if not npz_path.exists():
            prepare_audio(
                str(apath),
                self.audio_cfg["sample_rate"],
                self.audio_cfg["hop_ms"],
                self.audio_cfg["win_ms"],
                self.audio_cfg["n_mels"],
                self.audio_cfg["n_fft"],
                force=False,
            )
        spec = MelSpec.load_npz(str(npz_path))
        if self.use_spec_cache:
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

    def _filter_hit_objects(self, beatmap: Beatmap, start_ms: float, end_ms: float) -> List[object]:
        return [
            ho
            for ho in getattr(beatmap, "hit_objects", [])
            if start_ms <= float(getattr(ho, "time", 0.0)) < end_ms
        ]

    def _map_types_match(self, events: Sequence[Tuple[float, float, float]]) -> bool:
        if not self.map_types:
            return True

        requested = {typ for typ in self.map_types if typ in {"aim", "stream"}}
        if not requested:
            return True

        is_stream = self._is_stream_map(events)
        is_aim = self._is_aim_map(events)

        if "aim" in requested and "stream" in requested:
            return is_aim or is_stream
        if "aim" in requested:
            return is_aim and not is_stream
        if "stream" in requested:
            return is_stream and not is_aim
        return True

    def _is_stream_map(self, events: Sequence[Tuple[float, float, float]]) -> bool:
        if len(events) < 3:
            return False
        arr = np.asarray(events, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return False
        order = np.argsort(arr[:, 0])
        times = arr[order, 0]
        coords = arr[order, 1:3]
        intervals = np.diff(times)
        if intervals.size == 0:
            return False
        deltas = np.diff(coords, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        close_time = intervals <= self.stream_interval_ms
        close_space = distances <= self.stream_gap_distance
        stream_pairs = close_time & close_space
        if stream_pairs.size == 0:
            return False
        ratio = float(stream_pairs.sum()) / float(stream_pairs.size)
        return ratio >= self.stream_ratio

    def _is_aim_map(self, events: Sequence[Tuple[float, float, float]]) -> bool:
        if len(events) < 2:
            return False
        coords = np.array([(event[1], event[2]) for event in events], dtype=np.float32)
        if coords.shape[0] < 2:
            return False
        diffs = np.diff(coords, axis=0)
        if diffs.size == 0:
            return False
        dists = np.linalg.norm(diffs, axis=1)
        long_jumps = (dists >= self.aim_jump_distance).sum()
        ratio = float(long_jumps) / float(dists.size)
        return ratio >= self.aim_ratio

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

    def _cache_metadata(self) -> Dict[str, object]:
        return {
            "version": 5,
            "split": self.split,
            "context_beats": self.context_beats,
            "target_beats": self.target_beats,
            "ticks_per_beat": self.ticks_per_beat,
            "seq_len": self.seq_len,
            "attr_sizes": self.attr_sizes,
            "position_bin_size": self.data_cfg.get("position_bin_size"),
            "max_slider_ticks": self.data_cfg.get("max_slider_ticks"),
            "max_slides": self.data_cfg.get("max_slides"),
            "slider_sv_precision": self.data_cfg.get("slider_sv_precision"),
            "slider_sv_max": self.data_cfg.get("slider_sv_max"),
            "tick_tolerance_ms": self.tick_tolerance_ms,
            "map_types": sorted(self.map_types),
            "require_constant_bpm": self.require_constant_bpm,
            "require_tick_alignment": self.require_tick_alignment,
        }

    def _cache_meta_matches(self, meta: Dict[str, object]) -> bool:
        expected = self._cache_metadata()
        for key, value in expected.items():
            if meta.get(key) != value:
                return False
        return True

    def _save_cache(self, path: Path) -> None:
        if not self.samples:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(self._cache_metadata(), sort_keys=True)
        audio_paths = np.array([desc.audio_path for desc in self.samples], dtype=np.str_)
        start_frames = np.array([desc.start_frame for desc in self.samples], dtype=np.int32)
        frames_per_chunk = np.array([desc.frames_per_chunk for desc in self.samples], dtype=np.int32)
        frame_strides = np.array([desc.frame_stride for desc in self.samples], dtype=np.int16)
        raw_frames = np.array([desc.raw_frames for desc in self.samples], dtype=np.int32)
        token_lengths = np.array([desc.token_length for desc in self.samples], dtype=np.int32)
        tokens = np.stack([desc.tokens for desc in self.samples], axis=0).astype(np.int32)
        loss_masks = np.stack([desc.loss_mask for desc in self.samples], axis=0).astype(np.float32)
        np.savez_compressed(
            path,
            meta=np.array(meta),
            audio_paths=audio_paths,
            start_frames=start_frames,
            frames_per_chunk=frames_per_chunk,
            frame_strides=frame_strides,
            raw_frames=raw_frames,
            token_lengths=token_lengths,
            tokens=tokens,
            loss_masks=loss_masks,
        )

    def _load_cache(self, path: Path) -> None:
        data = np.load(path, allow_pickle=False)
        raw_meta = data["meta"]
        meta = json.loads(str(raw_meta.item() if raw_meta.shape == () else raw_meta))
        if not self._cache_meta_matches(meta):
            raise RuntimeError("cache metadata mismatch")

        audio_paths = data["audio_paths"].astype(str).tolist()
        start_frames = data["start_frames"]
        frames_per_chunk = data["frames_per_chunk"]
        frame_strides = data["frame_strides"]
        raw_frames = data["raw_frames"]
        token_lengths = data["token_lengths"]
        tokens = data["tokens"]
        loss_masks = data.get("loss_masks")
        if loss_masks is None:
            loss_masks = np.zeros((len(audio_paths), self.seq_len), dtype=np.float32)

        self.samples = []
        for idx, audio_path in enumerate(audio_paths):
            self.samples.append(
                SampleDescriptor(
                    audio_path=audio_path,
                    start_frame=int(start_frames[idx]),
                    frames_per_chunk=int(frames_per_chunk[idx]),
                    frame_stride=int(frame_strides[idx]),
                    raw_frames=int(raw_frames[idx]),
                    token_length=int(token_lengths[idx]),
                    tokens=np.array(tokens[idx], dtype=np.int64),
                    loss_mask=np.array(loss_masks[idx], dtype=np.float32),
                )
            )

    def _filter_events(
        self,
        events: Sequence[Tuple[float, float, float]],
        start_ms: float,
        end_ms: float,
    ) -> List[Tuple[float, float, float]]:
        return [event for event in events if start_ms <= event[0] < end_ms]

    def _limit_token_count(self, tokens: List[List[int]], limit: int) -> List[List[int]]:
        if limit <= 0:
            return []
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

    def _encode_chunk_tokens(
        self,
        hit_objects: Sequence[object],
        chunk_start_ms: float,
        ticks_per_sample: int,
        tick_duration_ms: float,
        beatmap: Beatmap,
    ) -> Tuple[List[List[int]], int, np.ndarray] | None:
        if ticks_per_sample <= 0:
            return None
        if not hit_objects and self.skip_empty:
            return None

        tokens = self.tokenizer.tokenize(
            hit_objects,
            chunk_start_ms=chunk_start_ms,
            tick_duration_ms=tick_duration_ms,
            max_ticks=ticks_per_sample,
            slider_sv_lookup=lambda slider: self._effective_slider_sv(beatmap, float(getattr(slider, "time", 0.0))),
            tick_tolerance_ms=self.tick_tolerance_ms,
        )
        max_tokens = max(1, ticks_per_sample - 1)
        tokens = self._limit_token_count(tokens, max_tokens)
        if not tokens and self.skip_empty:
            return None

        loss_mask: List[int] = [
            1 if self.tokenizer.tick_from_id(token[TokenAttr.TICK]) >= self.context_ticks else 0 for token in tokens
        ]

        tokens.append(self.tokenizer.eos_token())
        loss_mask.append(0)
        token_length = min(len(tokens), ticks_per_sample)
        tokens = tokens[:token_length]
        if tokens[-1][TokenAttr.TYPE] != TokenType.EOS:
            tokens[-1] = self.tokenizer.eos_token()
            loss_mask[-1] = 0

        while len(tokens) < ticks_per_sample:
            tokens.append(self.tokenizer.pad_token())
            loss_mask.append(0)

        loss_arr = np.asarray(loss_mask, dtype=np.float32)
        return tokens, token_length, loss_arr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        descriptor = self.samples[idx]
        spec = self._load_mel_spec(descriptor.audio_path)

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

        tokens = torch.from_numpy(descriptor.tokens).long()
        loss_mask = torch.from_numpy(descriptor.loss_mask.astype(np.float32)).bool()

        return {
            "audio": mel,
            "audio_length": audio_length,
            "tokens": tokens,
            "token_length": descriptor.token_length,
            "loss_mask": loss_mask,
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
    loss_mask_batch = torch.zeros(batch_size, max_token_len, dtype=torch.bool)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    token_lengths = torch.zeros(batch_size, dtype=torch.long)

    for idx, item in enumerate(batch):
        audio = item["audio"]
        tokens = item["tokens"]
        loss_mask = item["loss_mask"]

        audio_batch[idx, : audio.shape[0]] = audio
        token_batch[idx, : tokens.shape[0]] = tokens
        loss_mask_batch[idx, : loss_mask.shape[0]] = loss_mask
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
        "loss_mask": loss_mask_batch,
    }
