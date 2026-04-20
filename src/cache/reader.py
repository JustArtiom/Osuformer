from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .paths import CachePaths


@dataclass(frozen=True)
class AudioEntry:
    audio_key: str
    byte_offset: int
    byte_length: int
    n_frames: int
    n_mels: int


@dataclass(frozen=True)
class MapIndexEntry:
    beatmap_id: int
    slot_offset: int
    n_events: int
    metadata: dict[str, Any]


class CacheReader:
    def __init__(self, cache_root: Path, name: str, preload: bool = False):
        self.paths = CachePaths(root=cache_root / name)
        if not self.paths.audio_bin.exists():
            raise FileNotFoundError(f"audio.bin not found at {self.paths.audio_bin}")
        self._preload = preload
        self._audio_buffer = self._open_buffer(self.paths.audio_bin, np.uint8, preload)
        self._audio_index = self._load_audio_index()

        self._maps_buffer: np.ndarray | None = None
        self._map_index: dict[int, MapIndexEntry] = {}
        if self.paths.maps_index.exists() and self.paths.maps_bin.exists():
            self._maps_buffer = self._open_buffer(self.paths.maps_bin, np.int32, preload)
            self._map_index = self._load_maps_index()
        elif self.paths.maps.exists():
            raise RuntimeError(
                f"{self.paths.maps} exists but new binary format missing. "
                f"Run `python scripts/migrate_maps_to_binary.py --name {name}` first."
            )

    @staticmethod
    def _open_buffer(path: Path, dtype: type, preload: bool) -> np.ndarray:
        if preload:
            with open(path, "rb") as f:
                raw = f.read()
            return np.frombuffer(raw, dtype=dtype)
        return np.memmap(path, dtype=dtype, mode="r")

    def audio_keys(self) -> list[str]:
        return list(self._audio_index.keys())

    def map_ids(self) -> list[int]:
        return list(self._map_index.keys())

    def load_audio(self, audio_key: str) -> np.ndarray:
        entry = self._audio_index[audio_key]
        buf = self._audio_buffer[entry.byte_offset : entry.byte_offset + entry.byte_length]
        arr = np.frombuffer(buf, dtype=np.float16).reshape(entry.n_frames, entry.n_mels)
        return arr

    def load_map(self, beatmap_id: int) -> dict[str, Any]:
        if self._maps_buffer is None:
            raise RuntimeError("maps.bin not loaded")
        entry = self._map_index[beatmap_id]
        slot_count = entry.n_events * 2
        flat = self._maps_buffer[entry.slot_offset : entry.slot_offset + slot_count]
        event_types = flat[0::2]
        event_values = flat[1::2]
        out = dict(entry.metadata)
        out["beatmap_id"] = entry.beatmap_id
        out["event_types"] = event_types
        out["event_values"] = event_values
        return out

    def _load_audio_index(self) -> dict[str, AudioEntry]:
        out: dict[str, AudioEntry] = {}
        if not self.paths.audio_index.exists():
            return out
        table = pq.read_table(self.paths.audio_index)
        data = table.to_pydict()
        keys = data["audio_key"]
        offsets = data["byte_offset"]
        lengths = data["byte_length"]
        n_frames = data["n_frames"]
        n_mels_all = data["n_mels"]
        for i, key in enumerate(keys):
            out[str(key)] = AudioEntry(
                audio_key=str(key),
                byte_offset=int(offsets[i]),
                byte_length=int(lengths[i]),
                n_frames=int(n_frames[i]),
                n_mels=int(n_mels_all[i]),
            )
        return out

    def _load_maps_index(self) -> dict[int, MapIndexEntry]:
        table = pq.read_table(self.paths.maps_index)
        data = table.to_pydict()
        ids = data.pop("beatmap_id")
        offsets = data.pop("slot_offset")
        n_events = data.pop("n_events")
        out: dict[int, MapIndexEntry] = {}
        for i, bm_id in enumerate(ids):
            metadata = {k: (v[i] if isinstance(v, list) else v) for k, v in data.items()}
            out[int(bm_id)] = MapIndexEntry(
                beatmap_id=int(bm_id),
                slot_offset=int(offsets[i]),
                n_events=int(n_events[i]),
                metadata=metadata,
            )
        return out
