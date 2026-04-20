from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BeatmapIndex:
    train: list[int]
    val: list[int]


def split_beatmap_ids(
    beatmap_ids: list[int],
    train_ratio: float,
    seed: int,
) -> BeatmapIndex:
    rng = random.Random(seed)
    shuffled = list(beatmap_ids)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return BeatmapIndex(train=shuffled[:split], val=shuffled[split:])
