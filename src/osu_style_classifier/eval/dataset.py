from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BeatmapsetSample:
    set_id: int
    directory: Path
    osu_files: list[Path]


def discover_beatmapsets(songs_root: Path) -> list[BeatmapsetSample]:
    samples: list[BeatmapsetSample] = []
    if not songs_root.exists():
        return samples
    for entry in songs_root.iterdir():
        if not entry.is_dir():
            continue
        set_id = _parse_set_id(entry.name)
        if set_id is None:
            continue
        osu_files = sorted(entry.glob("*.osu"))
        if not osu_files:
            continue
        samples.append(BeatmapsetSample(set_id=set_id, directory=entry, osu_files=osu_files))
    return samples


def sample_beatmapsets(songs_root: Path, count: int, seed: int = 42) -> list[BeatmapsetSample]:
    all_sets = discover_beatmapsets(songs_root)
    rng = random.Random(seed)
    if count >= len(all_sets):
        return all_sets
    return rng.sample(all_sets, count)


def _parse_set_id(folder_name: str) -> int | None:
    token = folder_name.split(" ", 1)[0]
    if not token.isdigit():
        return None
    return int(token)
