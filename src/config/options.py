import functools
from pathlib import Path
from typing import Any, Callable, TypeVar

import click

from .loader import load_config
from .schemas.app import AppConfig

F = TypeVar("F", bound=Callable[..., Any])

_CONFIG_GROUPS: dict[str, str] = {
    "model": "models",
    "audio": "audio",
    "tokenizer": "tokenizer",
    "dataset": "dataset",
    "training": "training",
    "paths": "paths",
}
_CONFIG_ROOT = "config"
_DEFAULT_CONFIG = "config.yaml"


def with_config(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        config_path: str = kwargs.pop("config_path")
        set_overrides: tuple[str, ...] = kwargs.pop("set_overrides", ())

        config_root = Path(config_path).resolve().parent
        section_overrides: dict[str, Path] = {}
        for group, directory in _CONFIG_GROUPS.items():
            name: str | None = kwargs.pop(f"config_{group}", None)
            if name is not None:
                section_overrides[group] = config_root / directory / f"{name}.yaml"

        cfg = load_config(
            config_path,
            section_overrides=section_overrides or None,
            dotlist=list(set_overrides) or None,
        )
        return fn(*args, cfg=cfg, **kwargs)

    wrapper.__click_params__ = list(getattr(wrapper, "__click_params__", []))  # type: ignore[attr-defined]

    click.option(
        "--config",
        "config_path",
        type=str,
        default=str(Path(_CONFIG_ROOT) / _DEFAULT_CONFIG),
        show_default=True,
        help="Path to main config file",
    )(wrapper)

    for group in _CONFIG_GROUPS:
        click.option(
            f"--config-{group}",
            f"config_{group}",
            type=str,
            default=None,
            help=f"Override {group} config name",
        )(wrapper)  # type: ignore[arg-type]

    click.option(
        "--set",
        "set_overrides",
        multiple=True,
        help="Override config values in key=value format",
    )(wrapper)

    return wrapper  # type: ignore[return-value]
