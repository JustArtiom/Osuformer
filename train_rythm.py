from utils import audio
import matplotlib.pyplot as plt
import numpy as np
import librosa

example_path = "trainmaps/template/audio.mp3"
from_bpm = 184
to_bpm = 180

y, sr = librosa.load(example_path)
waveform = audio.update_waveform_bpm(y, from_bpm, to_bpm)
spectrogram = audio.waveform_to_spectrogram(waveform, 256, 128)

print(spectrogram.shape)