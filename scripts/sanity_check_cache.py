from __future__ import annotations

from collections import Counter
from pathlib import Path

import click

from src.cache import CacheReader
from src.cache.metadata import read_metadata
from src.cache.paths import CachePaths
from src.config.loader import load_config
from src.osu_tokenizer import DESCRIPTOR_TAGS


@click.command()
@click.option("--name", required=True, type=str)
@click.option("--config-path", default="config/config.yaml", type=click.Path(path_type=Path))
@click.option("--top-tags", default=15, type=int)
def main(name: str, config_path: Path, top_tags: int) -> None:
    cfg = load_config(str(config_path))
    cache_root = Path(cfg.paths.cache)
    paths = CachePaths(root=cache_root / name)

    reader = CacheReader(cache_root=cache_root, name=name)
    audio_keys = reader.audio_keys()
    map_ids = reader.map_ids()

    print("=== cache summary ===")
    print(f"  unique audios : {len(audio_keys)}")
    print(f"  maps          : {len(map_ids)}")

    durations = []
    event_counts = []
    bpms = []
    cs_values = []
    ar_values = []
    for bm_id in map_ids:
        rec = reader.load_map(bm_id)
        durations.append(float(rec["duration_ms"]) / 1000.0)
        event_counts.append(len(rec["event_types"]))
        bpms.append(float(rec["primary_bpm"]))
        cs_values.append(float(rec["circle_size"]))
        ar_values.append(float(rec["approach_rate"]))

    def stats(values: list[float], label: str, fmt: str = ".2f") -> None:
        if not values:
            print(f"  {label:<14}: empty")
            return
        sorted_v = sorted(values)
        lo = sorted_v[0]
        hi = sorted_v[-1]
        median = sorted_v[len(sorted_v) // 2]
        mean = sum(values) / len(values)
        print(f"  {label:<14}: min={lo:{fmt}}  median={median:{fmt}}  mean={mean:{fmt}}  max={hi:{fmt}}")

    print("\n=== per-map stats ===")
    stats(durations, "duration_s", ".1f")
    stats([float(e) for e in event_counts], "events", ".0f")
    stats(bpms, "primary_bpm", ".1f")
    stats(cs_values, "circle_size")
    stats(ar_values, "approach_rate")

    meta = read_metadata(paths)
    print("\n=== api metadata ===")
    print(f"  metadata rows : {len(meta)} ({100*len(meta)/max(1,len(map_ids)):.1f}% coverage)")
    if not meta:
        print("  (no metadata — run fetch_api_metadata.py)")
        return

    stars = [m.star_rating for m in meta.values()]
    years = [m.ranked_year for m in meta.values() if m.ranked_year > 0]
    tagged = sum(1 for m in meta.values() if m.descriptor_indices)

    stats(stars, "star_rating")
    if years:
        print(f"  years         : {min(years)}..{max(years)}")
    print(f"  with tags     : {tagged} ({100*tagged/len(meta):.1f}%)")

    tag_counts: Counter[str] = Counter()
    for m in meta.values():
        for idx in m.descriptor_indices:
            if 0 <= idx < len(DESCRIPTOR_TAGS):
                tag_counts[DESCRIPTOR_TAGS[idx]] += 1
    print(f"\n=== top {top_tags} tags ===")
    for tag, count in tag_counts.most_common(top_tags):
        print(f"  {count:>6d}  {tag}")

    star_bins = [0, 2, 3, 4, 5, 6, 7, 8, 10, 100]
    bin_labels = ["<2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-10", "10+"]
    star_counts = [0] * (len(star_bins) - 1)
    for s in stars:
        for i in range(len(star_bins) - 1):
            if star_bins[i] <= s < star_bins[i + 1]:
                star_counts[i] += 1
                break
    print("\n=== star distribution ===")
    total = sum(star_counts)
    for label, count in zip(bin_labels, star_counts):
        pct = 100 * count / max(1, total)
        bar = "#" * int(pct / 2)
        print(f"  {label:>5}★  {count:>6d}  {pct:>5.1f}%  {bar}")


if __name__ == "__main__":
    main()
