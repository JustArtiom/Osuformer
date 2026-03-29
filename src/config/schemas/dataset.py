from typing import TypedDict


class DatasetFiltersConfig(TypedDict):
    gamemode: list[int]
    max_difficulty: float
    min_difficulty: float


class DatasetConfig(TypedDict):
    version: str
    train_split: float
    val_split: float
    filters: DatasetFiltersConfig
