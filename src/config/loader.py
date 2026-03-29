from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


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


def load_config(
    config_path: str | Path,
    section_overrides: Mapping[str, str | Path] | None = None,
    dotlist: list[str] | None = None,
) -> DictConfig:
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

    OmegaConf.set_readonly(cfg, True)
    return cfg
