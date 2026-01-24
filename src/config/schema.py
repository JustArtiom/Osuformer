from __future__ import annotations
from typing import Union, Literal
from dataclasses import dataclass, field

@dataclass
class DatasetSplitConfig:
  train: float = 0.8
  val: float = 0.2

@dataclass
class DatasetMapFilterConfig:
  sr_min: int = 0
  sr_max: int = 11
  variable_bpm: bool = False
  tokenizer_limits: bool = True

@dataclass
class DatasetConfig:
  path: str = "dataset"
  split: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)
  map_filters: DatasetMapFilterConfig = field(default_factory=DatasetMapFilterConfig)
  workers: int = 1
  window_ms: int = 8000
  overlap: float = 0.9

@dataclass
class CacheConfig:
  path: str = "cache"

@dataclass
class CheckpointConfig:
  path: str = "checkpoints"

@dataclass
class AudioConfig:
  sample_rate: int = 22050
  hop_ms: int = 10
  win_ms: int = 25
  n_mels: int = 80
  n_fft: int = 1024
  normalize: bool = True

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
class TrainingEarlyStopConfig:
  patience: int = 10
  delta: float = 0.0001

@dataclass
class LRSchedulerConfig:
  factor: float = 0.5
  patience: int = 2
  threshold: float = 0.0001
  min_lr: float = 1e-6

@dataclass
class TrainingConfig:
  batch_size: int = 16
  epochs: int = 100
  lr: float = 0.0001
  use_ram: bool = True
  workers: int = 4
  early_stop: TrainingEarlyStopConfig = field(default_factory=TrainingEarlyStopConfig)
  lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)

@dataclass
class RootConfig:
  dataset: DatasetConfig = field(default_factory=DatasetConfig)
  cache: CacheConfig = field(default_factory=CacheConfig)
  checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
  audio: AudioConfig = field(default_factory=AudioConfig)
  tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
  model: ModelConfig = field(default_factory=ModelConfig)
  training: TrainingConfig = field(default_factory=TrainingConfig)

@dataclass
class ExperimentConfig(RootConfig):
  pass
