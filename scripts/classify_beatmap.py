from __future__ import annotations

from pathlib import Path

import click

from src.osu.beatmap import Beatmap
from src.osu_style_classifier import ClassifierConfig, classify


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--min-confidence", default=0.5, type=float)
@click.option("--top", default=0, type=int, help="Show only top N by confidence (0 = all).")
def main(path: Path, min_confidence: float, top: int) -> None:
    config = ClassifierConfig(min_confidence=min_confidence)
    beatmap = Beatmap(file_path=str(path))
    result = classify(beatmap, config)
    predictions = result.predictions
    if top > 0:
        predictions = predictions[:top]
    if not predictions:
        print("(no tags detected)")
        return
    for p in predictions:
        print(f"  {p.confidence:>4.2f}  {p.tag}")


if __name__ == "__main__":
    main()
