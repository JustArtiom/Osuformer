from __future__ import annotations

import os
import shutil
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
@click.option(
    "--reuse-audio-from",
    default=None,
    type=str,
    help="Name of existing cache to hardlink audio.bin/audio_index.parquet/metadata.parquet from; skips mel recomputation.",
)
def main(
    name: str,
    songs: Path | None,
    cache_root: Path | None,
    config_path: Path,
    limit: int | None,
    reuse_audio_from: str | None,
) -> None:
    cfg = load_config(str(config_path))
    songs_root = songs if songs is not None else Path(cfg.paths.data)
    cache_dest = cache_root if cache_root is not None else Path(cfg.paths.cache)
    print(f"source songs : {songs_root}")
    print(f"cache root   : {cache_dest}")
    print(f"cache name   : {name}")
    if reuse_audio_from is not None:
        _reuse_audio_from(cache_dest, reuse_audio_from, name)
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


def _reuse_audio_from(cache_root: Path, source_name: str, target_name: str) -> None:
    source_dir = cache_root / source_name
    target_dir = cache_root / target_name
    if not source_dir.exists():
        raise SystemExit(f"source cache not found: {source_dir}")
    if source_name == target_name:
        raise SystemExit("--reuse-audio-from must differ from --name")
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("audio.bin", "audio_index.parquet", "metadata.parquet"):
        src = source_dir / filename
        dst = target_dir / filename
        if not src.exists():
            continue
        if dst.exists():
            print(f"  skip reuse: {dst} already exists")
            continue
        try:
            os.link(src, dst)
            print(f"  hardlinked: {filename}")
        except OSError:
            print(f"  hardlink failed, copying: {filename}")
            shutil.copy2(src, dst)
    for stale in ("maps.bin", "maps_index.parquet", "maps.parquet"):
        p = target_dir / stale
        if p.exists():
            p.unlink()
            print(f"  cleared stale: {stale}")


if __name__ == "__main__":
    main()
