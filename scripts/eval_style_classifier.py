from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv

from src.osu_api import OsuClient
from src.osu_style_classifier import ClassifierConfig
from src.osu_style_classifier.eval import ApiLabelSource, evaluate, sample_beatmapsets


@click.command()
@click.option("--songs", default="/Volumes/MySSD/Songs", type=click.Path(path_type=Path))
@click.option("--count", default=100, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--cache-dir", default=".cache/osu_beatmapsets", type=click.Path(path_type=Path))
def main(songs: Path, count: int, seed: int, verbose: bool, cache_dir: Path) -> None:
    load_dotenv()
    client = OsuClient()
    labels = ApiLabelSource(client, cache_dir=cache_dir)
    config = ClassifierConfig()

    samples = sample_beatmapsets(songs, count=count, seed=seed)
    print(f"Evaluating {len(samples)} beatmapsets from {songs}...")
    report = evaluate(samples, labels, config, verbose=verbose, progress=True)
    print()
    print(report.summary())


if __name__ == "__main__":
    main()
