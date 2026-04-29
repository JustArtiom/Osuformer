from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MapperVocab:
    creator_to_idx: dict[str, int]
    top_n: int

    def encode(self, creator: str) -> int:
        return self.creator_to_idx.get(creator, 0)

    def __len__(self) -> int:
        return self.top_n + 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"creator_to_idx": self.creator_to_idx, "top_n": self.top_n}))

    @classmethod
    def load(cls, path: Path) -> "MapperVocab":
        data = json.loads(path.read_text())
        return cls(creator_to_idx=data["creator_to_idx"], top_n=int(data["top_n"]))


def build_mapper_vocab(creators: list[str], top_n: int = 1024) -> MapperVocab:
    counts = Counter(c for c in creators if c)
    most_common = counts.most_common(top_n)
    creator_to_idx = {creator: idx + 1 for idx, (creator, _) in enumerate(most_common)}
    return MapperVocab(creator_to_idx=creator_to_idx, top_n=top_n)
