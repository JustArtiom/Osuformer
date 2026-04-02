from dataclasses import asdict

import click
import yaml

from src.config.options import with_config
from src.config.schemas.app import AppConfig


@click.command()
@with_config
def main(cfg: AppConfig) -> None:
    print(yaml.dump(asdict(cfg), default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()
