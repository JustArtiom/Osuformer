from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.config.schemas.audio import AudioConfig
from src.config.schemas.model import ModelConfig

from .conformer import ConformerEncoder
from .decoder import TransformerDecoder


@dataclass
class OsuformerOutput:
    logits: Tensor
    encoder_out: Tensor


class Osuformer(nn.Module):
    def __init__(
        self,
        model_cfg: ModelConfig,
        audio_cfg: AudioConfig,
        vocab_size_in: int,
        vocab_size_out: int,
        max_decoder_len: int,
    ):
        super().__init__()
        total_ms = audio_cfg.context_ms + audio_cfg.generate_ms + audio_cfg.lookahead_ms
        max_audio_frames = int(total_ms / audio_cfg.hop_ms) + 16
        self.encoder = ConformerEncoder(model_cfg.encoder, n_mels=audio_cfg.n_mels, max_len=max_audio_frames)
        self.decoder = TransformerDecoder(model_cfg.decoder, vocab_size_in=vocab_size_in, max_len=max_decoder_len)
        if model_cfg.encoder.d_model != model_cfg.decoder.d_model:
            self.enc_to_dec = nn.Linear(model_cfg.encoder.d_model, model_cfg.decoder.d_model)
        else:
            self.enc_to_dec = nn.Identity()
        self.head = nn.Linear(model_cfg.decoder.d_model, vocab_size_out, bias=False)

    def forward(
        self,
        mel: Tensor,
        input_ids: Tensor,
        mel_key_padding_mask: Tensor | None = None,
        token_key_padding_mask: Tensor | None = None,
    ) -> OsuformerOutput:
        memory = self.encode(mel, mel_key_padding_mask=mel_key_padding_mask)
        logits = self.decode(
            input_ids,
            memory=memory,
            token_key_padding_mask=token_key_padding_mask,
            memory_key_padding_mask=mel_key_padding_mask,
        )
        return OsuformerOutput(logits=logits, encoder_out=memory)

    def encode(self, mel: Tensor, mel_key_padding_mask: Tensor | None = None) -> Tensor:
        memory = self.encoder(mel, key_padding_mask=mel_key_padding_mask)
        return self.enc_to_dec(memory)

    def decode(
        self,
        input_ids: Tensor,
        memory: Tensor,
        token_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        dec = self.decoder(
            input_ids,
            memory=memory,
            tgt_key_padding_mask=token_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.head(dec)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
