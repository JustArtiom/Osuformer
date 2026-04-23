from __future__ import annotations

from torch import nn

from src.config.schemas.audio import AudioConfig
from src.config.schemas.model import EncoderConfig

from .base import AudioEncoder
from .conformer_scratch import ConformerScratchEncoder


def build_audio_encoder(
    encoder_cfg: EncoderConfig,
    audio_cfg: AudioConfig,
    max_len: int,
) -> nn.Module:
    if encoder_cfg.type == "conformer_scratch":
        return ConformerScratchEncoder(encoder_cfg, audio_cfg, max_len=max_len)
    if encoder_cfg.type == "musicfm":
        from .musicfm import MusicFMEncoder

        return MusicFMEncoder(encoder_cfg, audio_cfg)
    raise ValueError(f"unknown encoder type: {encoder_cfg.type!r}")


__all__ = ["AudioEncoder", "ConformerScratchEncoder", "build_audio_encoder"]
