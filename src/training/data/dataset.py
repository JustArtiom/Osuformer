from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.cache.metadata import MetadataRecord, read_metadata
from src.cache.paths import CachePaths
from src.cache.reader import CacheReader
from src.config.schemas.audio import AudioConfig
from src.config.schemas.tokenizer import TokenizerConfig
from src.osu_tokenizer import Vocab

from .sequence_builder import SequenceBuilder, SequenceSample


@dataclass
class OsuSample:
    mel: Tensor
    input_ids: Tensor
    target_ids: Tensor
    loss_mask: Tensor


class OsuDataset(Dataset[OsuSample]):
    def __init__(
        self,
        cache_root: Path,
        cache_name: str,
        beatmap_ids: list[int],
        vocab: Vocab,
        tokenizer_cfg: TokenizerConfig,
        audio_cfg: AudioConfig,
        max_decoder_len: int,
        history_event_count: int,
        epoch_length: int,
        seed: int,
        preload: bool = False,
        reader: CacheReader | None = None,
    ):
        self._paths = CachePaths(root=cache_root / cache_name)
        self._reader = reader if reader is not None else CacheReader(cache_root=cache_root, name=cache_name, preload=preload)
        self._metadata: dict[int, MetadataRecord] = read_metadata(self._paths)
        available = set(self._reader.map_ids())
        self._beatmap_ids = [bid for bid in beatmap_ids if bid in available]
        self._vocab = vocab
        self._tokenizer_cfg = tokenizer_cfg
        self._audio_cfg = audio_cfg
        self._total_ms = tokenizer_cfg.context_ms + tokenizer_cfg.generate_ms + tokenizer_cfg.lookahead_ms
        self._frames_per_window = int(self._total_ms / audio_cfg.hop_ms)
        self._epoch_length = epoch_length
        self._builder = SequenceBuilder(
            vocab=vocab,
            tokenizer_cfg=tokenizer_cfg,
            max_len=max_decoder_len,
            history_event_count=history_event_count,
        )
        self._base_seed = seed

    def __len__(self) -> int:
        return self._epoch_length

    def __getitem__(self, index: int) -> OsuSample:
        bm_id = random.choice(self._beatmap_ids)
        map_rec = self._reader.load_map(bm_id)
        duration_ms = float(map_rec["duration_ms"])
        max_start = max(0.0, duration_ms - self._total_ms)
        window_start_ms = random.uniform(0.0, max_start)
        sample = self._build_sample(map_rec, window_start_ms)
        return sample

    def _build_sample(self, map_rec: dict, window_start_ms: float) -> OsuSample:
        metadata = self._metadata.get(int(map_rec["beatmap_id"]))
        seq = self._builder.build(
            event_types=list(map_rec["event_types"]),
            event_values=list(map_rec["event_values"]),
            map_record=map_rec,
            metadata=metadata,
            window_start_ms=window_start_ms,
        )
        mel = self._slice_mel(str(map_rec["audio_key"]), window_start_ms)
        return OsuSample(
            mel=mel,
            input_ids=seq.input_ids,
            target_ids=seq.target_ids,
            loss_mask=seq.loss_mask,
        )

    def _slice_mel(self, audio_key: str, window_start_ms: float) -> Tensor:
        full = self._reader.load_audio(audio_key)
        start_frame = int(round(window_start_ms / self._audio_cfg.hop_ms))
        end_frame = start_frame + self._frames_per_window
        window = full[start_frame:end_frame]
        if window.shape[0] < self._frames_per_window:
            pad = np.zeros((self._frames_per_window - window.shape[0], window.shape[1]), dtype=np.float16)
            window = np.concatenate([window, pad], axis=0)
        return torch.from_numpy(window.astype(np.float32))
