from typing import List, Union
from pathlib import Path
import librosa
import numpy as np
import warnings
import json
warnings.filterwarnings("ignore")

class StreamingAudioStats:
  def __init__(self):
    self.total_sum = 0.0
    self.total_sumsq = 0.0
    self.total_count = 0

  def update(self, mel: np.ndarray):
    mel = mel.astype(np.float64, copy=False)
    self.total_sum += mel.sum()
    self.total_sumsq += np.square(mel).sum()
    self.total_count += mel.size

  def finalize(self):
    if self.total_count == 0:
        return 0.0, 1.0

    mean = self.total_sum / self.total_count
    var = self.total_sumsq / self.total_count - mean * mean
    std = np.sqrt(max(var, 1e-8))
    return float(mean), float(std)

  def merge(self, other: "StreamingAudioStats"):
    self.total_sum += other.total_sum
    self.total_sumsq += other.total_sumsq
    self.total_count += other.total_count
  


def ms_to_samples(sample_rate, hop_ms):
  return int(sample_rate * hop_ms / 1000)

def audio_to_mel(path: Union[str, Path], sample_rate: float, hop_ms: float, win_ms: float, n_fft: int, n_mels: int):
  audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)
  duration_ms = len(audio) / sr * 1000

  hop_length = ms_to_samples(sample_rate, hop_ms)
  win_length = ms_to_samples(sample_rate, win_ms)

  mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    n_mels=n_mels,
    power=2.0
  )

  mel = librosa.power_to_db(mel, ref=np.max)

  return mel.astype(np.float32), duration_ms

def compute_mel_stats(mels: list[np.ndarray]):
  stacked = np.concatenate([m.flatten() for m in mels])
  return float(stacked.mean()), float(stacked.std())

def normalize_mel(mel: np.ndarray, mean: float, std: float):
  return (mel - mean) / (std + 1e-8)
