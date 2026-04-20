from __future__ import annotations

from pathlib import Path

import click
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

from src.cache.metadata import MetadataFetcher, read_metadata, write_metadata
from src.cache.paths import CachePaths
from src.config.loader import load_config
from src.osu_api import OsuClient


@click.command()
@click.option("--name", required=True, type=str)
@click.option("--cache-root", default=None, type=click.Path(path_type=Path), help="Override paths.cache from config.")
@click.option("--config-path", default="config/config.yaml", type=click.Path(path_type=Path))
@click.option("--api-cache-dir", default=".cache/osu_beatmapsets", type=click.Path(path_type=Path))
def main(name: str, cache_root: Path | None, config_path: Path, api_cache_dir: Path) -> None:
    load_dotenv()
    cfg = load_config(str(config_path))
    cache_dest = cache_root if cache_root is not None else Path(cfg.paths.cache)
    paths = CachePaths(root=cache_dest / name)
    if not paths.maps.exists():
        raise SystemExit(f"no maps.parquet at {paths.maps} — build cache first")

    already = read_metadata(paths)
    maps_table = pq.read_table(paths.maps, columns=["beatmap_id", "set_id"])
    rows = maps_table.to_pylist()
    by_set: dict[int, set[int]] = {}
    for r in rows:
        if int(r["beatmap_id"]) in already:
            continue
        by_set.setdefault(int(r["set_id"]), set()).add(int(r["beatmap_id"]))

    print(f"{len(already)} already fetched, {len(by_set)} sets remaining")
    if not by_set:
        return

    client = OsuClient()
    fetcher = MetadataFetcher(client, api_cache_dir)
    fetcher.load_tags()

    pending: list = []
    batch_size = 50
    for set_id, wanted in tqdm(sorted(by_set.items()), desc="sets"):
        records = fetcher.extract_records(set_id, wanted)
        pending.extend(records)
        if len(pending) >= batch_size:
            write_metadata(paths, pending)
            pending.clear()
    if pending:
        write_metadata(paths, pending)

    final = read_metadata(paths)
    print(f"metadata rows now: {len(final)}")


if __name__ == "__main__":
    main()
