from typing import Union
from pathlib import Path
import librosa
import numpy as np
import warnings
import json
warnings.filterwarnings("ignore")

class StreamingAudioStats:
  def __init__(self):
    self.count = 0
    self.mean = 0.0
    self.M2 = 0.0
    self.std = 0.0

  def update(self, x: np.ndarray):
    flat = x.flatten()
    for v in flat:
      self.count += 1
      delta = v - self.mean
      self.mean += delta / self.count
      delta2 = v - self.mean
      self.M2 += delta * delta2

  def finalize(self):
    if self.count < 2:
      return self.mean, 0.0
    variance = self.M2 / self.count
    return self.mean, np.sqrt(variance)
  
  def save(self, path: Union[str, Path]):
    mean, std = self.finalize()
    with open(path, "w") as f:
      json.dump({
        "count": int(self.count),
        "mean": float(mean),
        "M2": float(self.M2),
        "std": float(std)
      }, f, indent=2)

  def load(self, path: Union[str, Path]):
    with open(path, "r") as f:
      obj = json.load(f)
      self.count = obj["count"]
      self.mean = obj["mean"]
      self.M2 = obj["M2"]
      self.std = obj["std"]

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
