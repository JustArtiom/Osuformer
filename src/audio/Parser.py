# src/audio/spectrogram.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import librosa


@dataclass
class MelSpec:
  S_db: np.ndarray
  times: np.ndarray
  freqs: np.ndarray
  sr: int
  hop_length: int
  n_fft: int
  n_mels: int
  fmin: float
  fmax: Optional[float]
  power: float
  ref: float
  top_db: float
  frame_duration_ms: float
  window: str
  center: bool

  def to_figure(
    self,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "magma",
    show_colorbar: bool = True,
    ax=None,
    title: Optional[str] = None
  ):
    import matplotlib.pyplot as plt

    created_fig = False
    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)
      created_fig = True
    else:
      fig = ax.figure

    img = ax.imshow(
      self.S_db,
      aspect="auto",
      origin="lower",
      cmap=cmap,
      extent=[self.times[0], self.times[-1], self.freqs[0], self.freqs[-1]],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or "Mel Spectrogram")

    if show_colorbar:
      fig.colorbar(img, ax=ax, format="%+2.0f dB")

    return fig, ax, created_fig

  def show(self, **kwargs) -> None:
    import matplotlib.pyplot as plt

    fig, _, created_fig = self.to_figure(**kwargs)
    if created_fig:
      plt.show()
    else:
      fig.canvas.draw_idle()

  def save(
    self,
    path: str,
    *,
    dpi: int = 150,
    bbox_inches: str = "tight",
    close: bool = True,
    **kwargs
  ) -> None:
    import matplotlib.pyplot as plt

    fig, _, _ = self.to_figure(**kwargs)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    if close:
      plt.close(fig)

  def to_image_array(
    self,
    *,
    normalize: bool = True,
    dtype=np.uint8
  ) -> np.ndarray:
    data = np.array(self.S_db, copy=True)
    if normalize:
      data -= data.min()
      max_val = data.max()
      if max_val > 0:
        data /= max_val
    data = np.clip(data, 0.0, 1.0)
    return (data * (np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0)).astype(dtype)

def audio_to_mel_spectrogram(
  audio_path: str,
  *,
  sr: int = 22050,
  n_fft: int = 2048,
  hop_length: Optional[int] = None,
  hop_ms: Optional[float] = 10.0,
  win_length: Optional[int] = None,
  window: str = "hann",
  center: bool = True,
  pad_mode: str = "constant",
  
  n_mels: int = 128,
  fmin: float = 30.0,
  fmax: Optional[float] = None,
  
  power: float = 2.0,
  ref: float = 1.0,
  top_db: float = 80.0,
  
  mono: bool = True,
  offset: float = 0.0,
  duration: Optional[float] = None,
) -> MelSpec:
  y, sr_eff = librosa.load(audio_path, sr=sr, mono=mono, offset=offset, duration=duration)
  if hop_length is None:
    if hop_ms is None:
      hop_length = 512
    else:
      hop_length = max(1, int(round(sr_eff * (hop_ms / 1000.0))))
  S_mel = librosa.feature.melspectrogram(
    y=y, sr=sr_eff, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
    window=window, center=center, pad_mode=pad_mode,
    n_mels=n_mels, fmin=fmin, fmax=fmax, power=power
  )
  S_db = librosa.power_to_db(S_mel, ref=ref, top_db=top_db)
  times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr_eff, hop_length=hop_length, n_fft=n_fft)
  freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=(fmax or sr_eff / 2))
  return MelSpec(
    S_db=S_db, times=times, freqs=freqs, sr=sr_eff,
    hop_length=hop_length, n_fft=n_fft, n_mels=n_mels,
    fmin=fmin, fmax=fmax, power=power, ref=ref, top_db=top_db,
    frame_duration_ms=hop_length / sr_eff * 1000.0,
    window=window, center=center
  )
