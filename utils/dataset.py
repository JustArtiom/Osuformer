from dataclasses import dataclass
from typing import Callable
import numpy as np 
from .osu import Osu
from . import audio, args
import pandas as pd
import os

@dataclass
class DatasetCSVRecord:
    map_id: int
    difficulty_id: int
    difficulty_name: int
    difficulty_rating: float
    map_path: str
    audio_path: str
    length: int
    bpm: float
    user_rating: float


class Dataset:
    def __init__(self, config: args.TrainArgs):
        self.config = config
        self.info_path = os.path.join(config.dataset_path, "info.csv")
        self.maps_path = os.path.join(config.dataset_path, "maps")
        self.audios_path = os.path.join(config.dataset_path, "audios")
        self.dataset_info_df: pd.DataFrame = None 
        self.dataset_info_json: list[DatasetCSVRecord] = []

    def read_dataset_info(self):
        self.dataset_info_df = pd.read_csv(self.info_path)
        return self.dataset_info_df
        
    def filter_dataset_info(self):
        pass

    def parse_dataset_info(self, df: pd.DataFrame = None):
        df = df or self.dataset_info_df
        data = df.to_dict(orient='records')
        
        self.dataset_info_json = [
            DatasetCSVRecord(
                difficulty_id=d["DIFFICULTY ID"],
                difficulty_name=d["DIFFICULTY NAME"],
                difficulty_rating=d["DIFFICULTY RATING"],
                map_id=d["MAP ID"],
                map_path=d["MAP PATH"],
                audio_path=d["AUDIO PATH"],
                length=d["LENGTH"],
                bpm=d["BPM"],
                user_rating=d["USER RATING"]
            ) for d in data
        ]

        return self.dataset_info_json