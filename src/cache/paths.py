from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachePaths:
    root: Path

    @property
    def audio_bin(self) -> Path:
        return self.root / "audio.bin"

    @property
    def audio_index(self) -> Path:
        return self.root / "audio_index.parquet"

    @property
    def maps(self) -> Path:
        return self.root / "maps.parquet"

    @property
    def maps_bin(self) -> Path:
        return self.root / "maps.bin"

    @property
    def maps_index(self) -> Path:
        return self.root / "maps_index.parquet"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.parquet"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
