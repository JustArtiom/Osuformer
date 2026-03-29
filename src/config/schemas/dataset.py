from dataclasses import dataclass


@dataclass
class DatasetFiltersConfig:
    gamemode: list[int]
    max_difficulty: float
    min_difficulty: float


@dataclass
class DatasetConfig:
    version: str
    train_split: float
    val_split: float
    filters: DatasetFiltersConfig
