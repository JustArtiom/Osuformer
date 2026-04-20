from __future__ import annotations

import io
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .audio import AudioFeature
from .maps import MapRecord
from .paths import CachePaths


class AudioWriter:
    def __init__(self, paths: CachePaths):
        self._paths = paths
        self._bin_handle: io.BufferedWriter | None = None
        self._entries: list[dict[str, int | str]] = []
        self._existing_keys: set[str] = set()
        self._next_offset: int = 0
        if paths.audio_index.exists():
            table = pq.read_table(paths.audio_index)
            for row in table.to_pylist():
                self._entries.append(row)
                self._existing_keys.add(str(row["audio_key"]))
            if paths.audio_bin.exists():
                self._next_offset = paths.audio_bin.stat().st_size

    def __enter__(self) -> "AudioWriter":
        self._paths.ensure()
        self._bin_handle = open(self._paths.audio_bin, "ab")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._bin_handle is not None:
            self._bin_handle.close()
            self._bin_handle = None
        self.flush_index()

    def has(self, audio_key: str) -> bool:
        return audio_key in self._existing_keys

    def add(self, feature: AudioFeature) -> None:
        if feature.key in self._existing_keys:
            return
        if self._bin_handle is None:
            raise RuntimeError("AudioWriter must be used as a context manager")
        mel = feature.mel
        if mel.dtype != np.float16:
            mel = mel.astype(np.float16)
        assert mel.ndim == 2
        n_frames, n_mels = mel.shape
        byte_len = mel.nbytes
        self._bin_handle.write(mel.tobytes(order="C"))
        self._entries.append({
            "audio_key": feature.key,
            "byte_offset": int(self._next_offset),
            "byte_length": int(byte_len),
            "n_frames": int(n_frames),
            "n_mels": int(n_mels),
            "source": str(feature.source_path),
        })
        self._existing_keys.add(feature.key)
        self._next_offset += byte_len

    def flush_index(self) -> None:
        if not self._entries:
            return
        table = pa.Table.from_pylist(self._entries)
        pq.write_table(table, self._paths.audio_index)


class MapsWriter:
    def __init__(self, paths: CachePaths):
        self._paths = paths
        self._index_rows: list[dict] = []
        self._existing_ids: set[int] = set()
        self._bin_handle: io.BufferedWriter | None = None
        self._next_slot: int = 0
        if paths.maps_index.exists():
            table = pq.read_table(paths.maps_index)
            for row in table.to_pylist():
                self._index_rows.append(row)
                self._existing_ids.add(int(row["beatmap_id"]))
            if paths.maps_bin.exists():
                self._next_slot = paths.maps_bin.stat().st_size // 4

    def __enter__(self) -> "MapsWriter":
        self._paths.ensure()
        self._bin_handle = open(self._paths.maps_bin, "ab")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._bin_handle is not None:
            self._bin_handle.close()
            self._bin_handle = None
        self.flush()

    def has(self, beatmap_id: int) -> bool:
        return beatmap_id in self._existing_ids

    def add(self, record: MapRecord) -> None:
        if record.beatmap_id in self._existing_ids:
            return
        if self._bin_handle is None:
            raise RuntimeError("MapsWriter must be used as a context manager")
        types = np.asarray(record.event_types, dtype=np.int32)
        values = np.asarray(record.event_values, dtype=np.int32)
        assert types.shape == values.shape
        flat = np.empty(types.shape[0] * 2, dtype=np.int32)
        flat[0::2] = types
        flat[1::2] = values
        self._bin_handle.write(flat.tobytes(order="C"))

        row = asdict(record)
        row.pop("event_types", None)
        row.pop("event_values", None)
        row["slot_offset"] = int(self._next_slot)
        row["n_events"] = int(types.shape[0])
        self._index_rows.append(row)
        self._existing_ids.add(record.beatmap_id)
        self._next_slot += flat.shape[0]

    def flush(self) -> None:
        if not self._index_rows:
            return
        table = pa.Table.from_pylist(self._index_rows)
        pq.write_table(table, self._paths.maps_index)
