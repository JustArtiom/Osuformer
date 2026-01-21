from __future__ import annotations
import click

from pathlib import Path
from typing import Optional, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from .schema import ExperimentConfig


def _resolve_size_path(config_path: Path, size: str) -> Path:
  base_dir = config_path.parent
  candidate = base_dir / "size" / f"{size}.yaml"
  if candidate.exists():
    return candidate
  alt = base_dir / f"{size}.yaml"
  if alt.exists():
    return alt
  raise FileNotFoundError(f"Size config not found for '{size}' (looked in {candidate} and {alt})")

def _merge_configs(base: Union[DictConfig, ListConfig], override: Optional[Union[DictConfig, ListConfig]]) -> Union[ListConfig, DictConfig]:
  if override is None:
    return base
  merged = OmegaConf.merge(base, override)
  return merged

def load_config(config_path: str, size: Optional[str] = None) -> ExperimentConfig:
  path = Path(config_path)
  if not path.exists():
    alt = Path(str(config_path).replace("configs/", "config/"))
    if alt.exists():
      path = alt
    else:
      raise FileNotFoundError(f"Config file not found: {path}")

  base_cfg: Union[DictConfig, ListConfig] = OmegaConf.load(str(path))
  override_cfg: Optional[Union[DictConfig, ListConfig]] = None

  if size:
    size_path = _resolve_size_path(path, size)
    override_cfg = OmegaConf.load(size_path)

  merged = _merge_configs(base_cfg, override_cfg)
  structured = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), merged)
  config: ExperimentConfig = cast(ExperimentConfig, OmegaConf.to_object(structured))
  return config

def config_options(fn):
  import functools

  @click.option(
    "--config",
    "config_path",
    type=str,
    default="configs/default.yaml",
    help="Path to config file"
  )
  @click.option(
    "--size",
    "size",
    type=str,
    default=None,
    help="Model size override (loads configs/size/<size>.yaml if exists)"
  )
  @functools.wraps(fn)
  def wrapper(*args, config_path: str, size: Optional[str], **kwargs):
    config = load_config(config_path, size)
    return fn(*args, config=config, **kwargs)

  return wrapper