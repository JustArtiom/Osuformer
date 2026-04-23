from __future__ import annotations

from torch import Tensor, nn

from src.config.schemas.audio import AudioConfig
from src.config.schemas.model import EncoderConfig

from .third_party.musicfm_25hz import MusicFM25Hz

_MUSICFM_NATIVE_DIM = 1024
_MUSICFM_TEMPORAL_SUBSAMPLE = 4


class MusicFMEncoder(nn.Module):
    def __init__(self, encoder_cfg: EncoderConfig, audio_cfg: AudioConfig) -> None:
        super().__init__()
        if audio_cfg.n_mels != 128:
            raise ValueError(
                f"MusicFM expects n_mels=128, got {audio_cfg.n_mels}; rebuild the cache with the v3-musicfm audio config."
            )
        self.musicfm = MusicFM25Hz(
            stat_path=encoder_cfg.musicfm_stats_path,
            model_path=encoder_cfg.musicfm_model_path,
        )
        del self.musicfm.preprocessor_melspec_2048
        del self.musicfm.linear
        del self.musicfm.cls_token
        for attr in list(self.musicfm._modules.keys()):
            if attr.startswith("quantizer_"):
                delattr(self.musicfm, attr)
        if hasattr(self.musicfm.conformer, "pos_conv_embed"):
            del self.musicfm.conformer.pos_conv_embed
        self.layer_ix = encoder_cfg.musicfm_layer
        if not 0 <= self.layer_ix <= 12:
            raise ValueError(f"musicfm_layer must be in [0, 12], got {self.layer_ix}")
        self._apply_freeze(encoder_cfg.freeze_first_n_layers)
        self.output_dim = _MUSICFM_NATIVE_DIM
        self.feature_rate_hz = 1000.0 / (audio_cfg.hop_ms * _MUSICFM_TEMPORAL_SUBSAMPLE)

    def _apply_freeze(self, freeze_first_n_layers: int) -> None:
        if freeze_first_n_layers <= 0:
            return
        for p in self.musicfm.conv.parameters():
            p.requires_grad = False
        layers = self.musicfm.conformer.layers
        n = min(freeze_first_n_layers, len(layers))
        for i in range(n):
            for p in layers[i].parameters():
                p.requires_grad = False

    def forward(self, mel: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        del key_padding_mask
        x = mel.transpose(1, 2)
        x = self.musicfm.conv(x)
        out = self.musicfm.conformer(x, output_hidden_states=True)
        hidden_states: tuple[Tensor, ...] = out["hidden_states"]
        return hidden_states[self.layer_ix]
