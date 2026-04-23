# Vendored from https://github.com/minzwon/musicfm
# MIT License - Copyright 2023 ByteDance Inc. (see LICENSE in this directory)
#
# Adapted: reimplemented with pure torch.stft + manual htk mel filterbank to
# drop the torchaudio dependency (which had ABI issues on our cluster).
# Output numerically matches torchaudio's
# MelSpectrogram(defaults) + AmplitudeToDB(stype="power", top_db=None).
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MelSTFT(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 2048,
        hop_length: int = 240,
        n_mels: int = 128,
        is_db: bool = False,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.is_db = is_db
        self.register_buffer("_window", torch.hann_window(n_fft, periodic=True), persistent=False)
        fb = _mel_filterbank_htk(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.register_buffer("_mel_fb", fb, persistent=False)

    def forward(self, waveform: Tensor) -> Tensor:
        window: Tensor = self._window  # type: ignore[assignment]
        mel_fb: Tensor = self._mel_fb  # type: ignore[assignment]
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window.to(waveform.dtype),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)
        mel = torch.matmul(mel_fb.to(power.dtype), power)
        if self.is_db:
            mel = 10.0 * torch.log10(mel.clamp(min=1e-10))
        return mel


def _mel_filterbank_htk(sample_rate: int, n_fft: int, n_mels: int) -> Tensor:
    f_max = sample_rate / 2.0
    mel_min = _hz_to_mel_htk(0.0)
    mel_max = _hz_to_mel_htk(f_max)
    mels = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz_htk(mels)
    fft_bin_hz = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    fb = torch.zeros(n_mels, fft_bin_hz.shape[0])
    for i in range(n_mels):
        left = hz_points[i].item()
        center = hz_points[i + 1].item()
        right = hz_points[i + 2].item()
        up = (fft_bin_hz - left) / max(center - left, 1e-10)
        down = (right - fft_bin_hz) / max(right - center, 1e-10)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


def _hz_to_mel_htk(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz_htk(mel: Tensor) -> Tensor:
    return 700.0 * (torch.pow(torch.tensor(10.0), mel / 2595.0) - 1.0)
