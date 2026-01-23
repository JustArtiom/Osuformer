import click

from .data import Dataset
from .config import config_options, ExperimentConfig

@click.command()
@click.argument("name", type=str)
@click.option("--limit", type=int, default=-1)
@click.option("--workers", type=int, default=None)
@config_options
def main(config: ExperimentConfig, name: str, limit: int, workers: int):
  dataset = Dataset(config, workers=workers)
  dataset.build_cache(name=name, limit=limit)


if __name__ == "__main__":
  main()