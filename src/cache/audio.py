from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from src.config.schemas.audio import AudioConfig


@dataclass(frozen=True)
class AudioFeature:
    key: str
    mel: np.ndarray
    source_path: Path


def hash_audio_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def compute_mel(path: Path, audio_cfg: AudioConfig) -> np.ndarray:
    if audio_cfg.preset == "musicfm":
        return _compute_mel_musicfm(path, audio_cfg)
    if audio_cfg.preset == "default":
        return _compute_mel_default(path, audio_cfg)
    raise ValueError(f"unknown audio preset: {audio_cfg.preset!r}")


def _compute_mel_default(path: Path, audio_cfg: AudioConfig) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != audio_cfg.sample_rate:
        mono = _resample_linear(mono, sr, audio_cfg.sample_rate)
    hop_length = int(round(audio_cfg.hop_ms * audio_cfg.sample_rate / 1000.0))
    win_length = int(round(audio_cfg.win_ms * audio_cfg.sample_rate / 1000.0))
    mel_basis = _mel_filterbank(
        sample_rate=audio_cfg.sample_rate,
        n_fft=audio_cfg.n_fft,
        n_mels=audio_cfg.n_mels,
    )
    power = _stft_power(mono, n_fft=audio_cfg.n_fft, hop_length=hop_length, win_length=win_length)
    mel = mel_basis @ power
    log_mel = np.log10(mel + 1e-10)
    return log_mel.T.astype(np.float16)


def _compute_mel_musicfm(path: Path, audio_cfg: AudioConfig) -> np.ndarray:
    import torch

    from src.model.encoders.third_party.musicfm_features import MelSTFT

    if audio_cfg.stats_path is None:
        raise ValueError("musicfm preset requires audio.stats_path pointing to msd_stats.json")
    mean, std = _load_musicfm_stats(audio_cfg.stats_path)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != audio_cfg.sample_rate:
        mono = _resample_linear(mono, sr, audio_cfg.sample_rate)
    extractor = _get_musicfm_extractor(audio_cfg.sample_rate, audio_cfg.n_fft, audio_cfg.hop_ms, audio_cfg.n_mels)
    with torch.no_grad():
        wav = torch.from_numpy(mono).unsqueeze(0)
        mel = extractor(wav)[..., :-1]
        mel = (mel - mean) / std
    arr = mel.squeeze(0).cpu().numpy()
    return arr.T.astype(np.float16)


@lru_cache(maxsize=4)
def _load_musicfm_stats(stats_path: str) -> tuple[float, float]:
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return float(stats["melspec_2048_mean"]), float(stats["melspec_2048_std"])


@lru_cache(maxsize=4)
def _get_musicfm_extractor(sample_rate: int, n_fft: int, hop_ms: int, n_mels: int):
    from src.model.encoders.third_party.musicfm_features import MelSTFT

    hop_length = int(round(hop_ms * sample_rate / 1000.0))
    return MelSTFT(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, is_db=True).eval()


def _resample_linear(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    duration = y.shape[0] / orig_sr
    new_len = int(round(duration * target_sr))
    old_idx = np.linspace(0, y.shape[0] - 1, num=new_len, dtype=np.float64)
    lo = np.floor(old_idx).astype(np.int64)
    hi = np.clip(lo + 1, 0, y.shape[0] - 1)
    frac = (old_idx - lo).astype(np.float32)
    return y[lo] * (1.0 - frac) + y[hi] * frac


def _stft_power(y: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    window = np.hanning(win_length).astype(np.float32)
    if win_length < n_fft:
        pad = n_fft - win_length
        left = pad // 2
        right = pad - left
        window = np.pad(window, (left, right))
    pad = n_fft // 2
    y_padded = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (y_padded.shape[0] - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y_padded,
        shape=(n_frames, n_fft),
        strides=(y_padded.strides[0] * hop_length, y_padded.strides[0]),
    ).copy()
    frames *= window
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return (np.abs(spec) ** 2).T.astype(np.float32)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    fmax = sample_rate / 2.0
    mel_min = _hz_to_mel(np.array(0.0))
    mel_max = _hz_to_mel(np.array(fmax))
    mels = np.linspace(float(mel_min), float(mel_max), n_mels + 2)
    hz_points = _mel_to_hz(mels)
    fft_bin_hz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    filterbank = np.zeros((n_mels, fft_bin_hz.shape[0]), dtype=np.float32)
    for i in range(n_mels):
        left = float(hz_points[i])
        center = float(hz_points[i + 1])
        right = float(hz_points[i + 2])
        up = (fft_bin_hz - left) / max(center - left, 1e-10)
        down = (right - fft_bin_hz) / max(right - center, 1e-10)
        filterbank[i] = np.maximum(0.0, np.minimum(up, down))
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    filterbank *= enorm[:, np.newaxis]
    return filterbank


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def compute_audio_feature(path: Path, audio_cfg: AudioConfig) -> AudioFeature:
    key = hash_audio_file(path)
    mel = compute_mel(path, audio_cfg)
    return AudioFeature(key=key, mel=mel, source_path=path)
