from dataclasses import dataclass

from .audio import AudioConfig
from .dataset import DatasetConfig
from .model import ModelConfig
from .paths import PathsConfig
from .tokenizer import TokenizerConfig
from .training import TrainingConfig


@dataclass
class AppConfig:
    model: ModelConfig
    audio: AudioConfig
    tokenizer: TokenizerConfig
    dataset: DatasetConfig
    training: TrainingConfig
    paths: PathsConfig
