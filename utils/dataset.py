import os
from typing import Callable
import numpy as np


class Dataset:
    def __init__(self, train_path: str):
        self.train_path = train_path

    """
    Loop trough an array and find the first file 
    meeting the file type in the parameters
    """

    def find_file_type(self, files: list[str], fileType: str) -> str:
        for file in files:
            if file.endswith("."+fileType):
                return file
        return None

    """
    This is a function that calls your callback to loop trough the 
    array of training folder if a mp3 file and a osu file exists
    """

    def loop_trough_train_files(self, cb: Callable[[str, str], None]) -> None:
        folders = [item for item in os.listdir(
            self.train_path) if os.path.isdir(os.path.join(self.train_path, item))]

        for folder_name in folders:
            mapPath = os.path.join(self.train_path, folder_name)
            files = os.listdir(mapPath)
            mp3 = self.find_file_type(files, "mp3")
            osu = self.find_file_type(files, "osu")

            full_mp3 = os.path.join(self.train_path, folder_name, mp3)
            full_osu = os.path.join(self.train_path, folder_name, osu)

            if (mp3 and osu):
                cb(full_mp3, full_osu)

            continue

        return None

    """
    Get a 2D numpy array of audio files and osu maps
    """

    def getFiles(self) -> tuple[list[str], list[str]]:
        mp3s = []
        osus = []
        self.loop_trough_train_files(
            lambda mp3, osu: (mp3s.append(mp3), osus.append(osu)))

        return mp3s, osus

    def get_data_prepared(
            self,
            audios_data,
            maps_beats,
            collect_range,
            maps_hit_objects,
            maps_intensity_data,
            maps_volume_data,
    ):
        X_audio = []
        X_features = []
        Y_labels = []
        Y_features = []

        for i in range(len(audios_data)):
            for o, beat in enumerate(maps_beats[i]):
                beat_index = int(beat)
                start_index = max(0, beat_index - collect_range)
                end_index = min(
                    len(audios_data[0]), beat_index + collect_range)

                segment_length = end_index - start_index

                segment = np.zeros((64, collect_range * 2))

                if segment_length > 0:
                    segment[:, collect_range - (beat_index - start_index):collect_range + (
                        end_index - beat_index)] = audios_data[i][:, start_index:end_index]

                X_audio.append(segment)
                X_features.append(
                    [maps_intensity_data[i][o-1], maps_volume_data[i][o-1]])
                Y_labels.append(maps_hit_objects[i][o][1:])
                Y_features.append(
                    [maps_intensity_data[i][o], maps_volume_data[i][o]])

        return np.array(X_audio), np.array(X_features), np.array(Y_labels), np.array(Y_features)
