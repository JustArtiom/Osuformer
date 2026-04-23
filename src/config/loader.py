from collections.abc import Mapping
from dataclasses import fields as dc_fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf

from .schemas.app import AppConfig

T = TypeVar("T")


def _resolve_base(config_path: Path) -> DictConfig:
    raw = OmegaConf.load(config_path)
    assert isinstance(raw, DictConfig)

    if "_base_" not in raw:
        return raw

    base_val = raw["_base_"]
    bases: list[str] = [base_val] if isinstance(base_val, str) else list(base_val)

    merged = OmegaConf.create({})
    assert isinstance(merged, DictConfig)
    for base_name in bases:
        base_cfg = _resolve_base(config_path.parent / base_name)
        result = OmegaConf.merge(merged, base_cfg)
        assert isinstance(result, DictConfig)
        merged = result

    keys = [k for k in raw if k != "_base_"]
    raw_no_base = OmegaConf.masked_copy(raw, keys)  # type: ignore[arg-type]
    final = OmegaConf.merge(merged, raw_no_base)
    assert isinstance(final, DictConfig)
    return final


def _from_dict(cls: type[T], data: Any) -> T:
    from typing import get_type_hints

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dc_fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        kwargs[f.name] = (
            _from_dict(hints[f.name], data[f.name]) if is_dataclass(hints[f.name]) else data[f.name]
        )
    return cls(**kwargs)


def load_config(
    config_path: str | Path,
    section_overrides: Mapping[str, str | Path] | None = None,
    dotlist: list[str] | None = None,
) -> AppConfig:
    cfg = _resolve_base(Path(config_path).resolve())

    if section_overrides:
        for section_path in section_overrides.values():
            override = _resolve_base(Path(section_path).resolve())
            merged = OmegaConf.merge(cfg, override)
            assert isinstance(merged, DictConfig)
            cfg = merged

    if dotlist:
        cli_cfg = OmegaConf.from_dotlist(dotlist)
        merged = OmegaConf.merge(cfg, cli_cfg)
        assert isinstance(merged, DictConfig)
        cfg = merged

    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    return _from_dict(AppConfig, container)
