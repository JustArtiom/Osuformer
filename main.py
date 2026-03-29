import click
from src.config import AppConfig, with_config


@click.command()
@with_config
def main(cfg: AppConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    main()