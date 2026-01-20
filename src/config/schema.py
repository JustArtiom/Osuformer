from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
  path: str = ""

@dataclass
class CacheConfig:
  path: str = ""

@dataclass
class TokenizerConfig:
  DT_BIN_MS: int = 10
  X_BINS: int = 32
  Y_BINS: int = 24
  SLIDER_CP_LIMIT: int = 9
  SLIDER_VEL_LIMIT: int = 5
  BPM_MIN: int = 30
  BPM_MAX: int = 300
  BPM_JUMP: int = 5
  SLIDER_LEN_BINS: int = 10
  SLIDER_LEN_MAX: int = 1000

@dataclass
class EncoderConfig:
  d_model: int = 768
  layers: int = 12
  heads: int = 12
  ffn_dim: int = 3072
  conv_kernel: int = 31
  dropout: float = 0.1

@dataclass
class DecoderConfig:
  d_model: int = 768
  layers: int = 10
  heads: int = 12
  ffn_dim: int = 3072
  dropout: float = 0.1


@dataclass
class ModelConfig:
  encoder: EncoderConfig = field(default_factory=EncoderConfig)
  decoder: DecoderConfig = field(default_factory=DecoderConfig)

@dataclass
class RootConfig:
  dataset: DatasetConfig = field(default_factory=DatasetConfig)
  cache: CacheConfig = field(default_factory=CacheConfig)
  tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
  model: ModelConfig = field(default_factory=ModelConfig)

@dataclass
class ExperimentConfig(RootConfig):
  pass
