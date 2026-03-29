import click
from omegaconf import DictConfig

from src.config import with_config


@click.command()
@with_config
def main(cfg: DictConfig) -> None:
    print(cfg)
    pass


if __name__ == "__main__":
    main()