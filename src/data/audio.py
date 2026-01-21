from typing import Union
from pathlib import Path
import librosa
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def hop_ms_to_samples(sample_rate, hop_ms):
  return int(sample_rate * hop_ms / 1000)

def audio_to_mel(path: Union[str, Path], sample_rate: float, hop_ms: float, win_ms: float, n_fft: int, n_mels: float):
  audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)

  hop_length = hop_ms_to_samples(sample_rate, hop_ms)
  win_length = hop_ms_to_samples(sample_rate, win_ms)

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

  return mel.astype(np.float32)

def compute_mel_stats(mels: list[np.ndarray]):
  stacked = np.concatenate([m.flatten() for m in mels])
  return float(stacked.mean()), float(stacked.std())

def normalize_mel(mel: np.ndarray, mean: float, std: float):
  return (mel - mean) / std

def audio_to_mel_norm(
  path: str, 
  sample_rate: float,
  hop_ms: float,
  win_ms: float,
  n_fft: int,
  n_mels: int,
  mean: float,
  std: float
):
  mel = audio_to_mel(
    path,
    sample_rate=sample_rate,
    hop_ms=hop_ms,
    win_ms=win_ms,
    n_fft=n_fft,
    n_mels=n_mels,
  )

  if mean is not None and std is not None:
    mel = (mel - mean) / std

  return mel