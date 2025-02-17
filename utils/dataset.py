from typing import Callable
import numpy as np 
import os 
from .osu import Osu
from . import audio, args

class Dataset:
    def __init__(self, config: args.Args):
        self.config = config
        

    def find_file_type(self, files: list[str], fileType: str) -> str:
        for file in files:
            if file.endswith("."+fileType):
                return file
        return None

    def loop_trough_train_files(self, train_path, cb: Callable[[str, str], None]) -> None:
        folders = [item for item in os.listdir(
            train_path) if os.path.isdir(os.path.join(train_path, item))]
        
        for folder_name in folders:
            mapPath = os.path.join(train_path, folder_name)
            files = os.listdir(mapPath)
            mp3 = self.find_file_type(files, "mp3")
            osu = self.find_file_type(files, "osu")
            full_mp3 = os.path.join(train_path, folder_name, mp3)
            full_osu = os.path.join(train_path, folder_name, osu)
            if (mp3 and osu):
                cb(full_mp3, full_osu)
            else:
                print(f"It seems i didnt find both files in the foilder {folder_name}")
                continue
        return None

    def getFiles(self,train_path: str) -> tuple[list[str], list[str]]:
        mp3s = []
        osus = []
        self.loop_trough_train_files(train_path,
            lambda mp3, osu: (mp3s.append(mp3), osus.append(osu)))

        return osus, mp3s

    def load(self):
        osus, mp3s = self.getFiles(self.config.train_path)

        osu_maps: list[Osu] = []
        audio_maps: list[np.ndarray] = []
        sr_last = None

        for i in range(0, max(len(osus), len(mp3s))):
            print("Loading: ", osus[i])
            osu_maps.append(Osu.from_file(osus[i]))
            data, sr = audio.load_audio(
                mp3s[i], 
                self.config.sample_rate, 
                self.config.n_mels, 
                self.config.frame_ms
            )

            if sr_last and sr != sr_last:
                print(f"Dataset load: Variable SR detected. old={sr_last} new={sr}")

            audio_maps.append(data)

        return osu_maps, audio_maps

    def generate_inputs(self, osu_maps: list[Osu]):
        inputs = []
        for osu in osu_maps:
            pass

        return inputs

