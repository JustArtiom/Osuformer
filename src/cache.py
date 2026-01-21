import click

from .data import Dataset
from .config import config_options, ExperimentConfig

@click.command()
@click.argument("name", type=str)
@click.option("--limit", type=int, default=-1)
@config_options
def main(config: ExperimentConfig, name: str, limit: int):
  dataset = Dataset(config)
  dataset.build_cache(name=name, limit=limit)


if __name__ == "__main__":
  main()