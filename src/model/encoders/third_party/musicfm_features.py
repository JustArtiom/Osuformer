# Vendored from https://github.com/minzwon/musicfm
# MIT License - Copyright 2023 ByteDance Inc. (see LICENSE in this directory)
from __future__ import annotations

import torchaudio
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
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.is_db = is_db
        if is_db:
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform: Tensor) -> Tensor:
        spec: Tensor = self.mel_stft(waveform)
        if self.is_db:
            spec = self.amplitude_to_db(spec)
        return spec
