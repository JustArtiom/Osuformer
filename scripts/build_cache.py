from __future__ import annotations

from pathlib import Path

import click

from src.cache import build_cache
from src.config.loader import load_config


@click.command()
@click.option("--name", required=True, type=str, help="Cache subdirectory name.")
@click.option("--songs", default=None, type=click.Path(path_type=Path), help="Override paths.data from config.")
@click.option("--cache-root", default=None, type=click.Path(path_type=Path), help="Override paths.cache from config.")
@click.option("--config-path", default="config/config.yaml", type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int, help="Optional cap on number of beatmapsets.")
def main(name: str, songs: Path | None, cache_root: Path | None, config_path: Path, limit: int | None) -> None:
    cfg = load_config(str(config_path))
    songs_root = songs if songs is not None else Path(cfg.paths.data)
    cache_dest = cache_root if cache_root is not None else Path(cfg.paths.cache)
    print(f"source songs : {songs_root}")
    print(f"cache root   : {cache_dest}")
    print(f"cache name   : {name}")
    stats = build_cache(
        songs_root=songs_root,
        cache_root=cache_dest,
        name=name,
        audio_cfg=cfg.audio,
        tokenizer_cfg=cfg.tokenizer,
        limit=limit,
    )
    print("=== cache build stats ===")
    print(f"  sets seen      : {stats.sets_seen}")
    print(f"  sets skipped   : {stats.sets_skipped}")
    print(f"  maps written   : {stats.maps_written}")
    print(f"  maps skipped   : {stats.maps_skipped}")
    print(f"  audios written : {stats.audios_written}")
    print(f"  audios reused  : {stats.audios_reused}")
    print(f"  errors         : {stats.errors}")


if __name__ == "__main__":
    main()
