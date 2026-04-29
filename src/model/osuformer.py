from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.config.schemas.audio import AudioConfig
from src.config.schemas.model import ModelConfig
from src.config.schemas.tokenizer import TokenizerConfig
from src.outline import SongOutliner, SongOutlinerConfig

from .aux_heads import AuxHeads, AuxOutputs, pool_encoder_output
from .conditioning import ConditionEncoder, ConditionFeatures, default_condition_spec
from .decoder import BlockCache, TransformerDecoder
from .encoders import build_audio_encoder


@dataclass
class OsuformerOutput:
    logits: Tensor
    encoder_out: Tensor
    aux: AuxOutputs


class Osuformer(nn.Module):
    def __init__(
        self,
        model_cfg: ModelConfig,
        audio_cfg: AudioConfig,
        tokenizer_cfg: TokenizerConfig,
        vocab_size_in: int,
        vocab_size_out: int,
        max_decoder_len: int,
    ):
        super().__init__()
        total_ms = audio_cfg.context_ms + audio_cfg.generate_ms + audio_cfg.lookahead_ms
        max_audio_frames = int(total_ms / audio_cfg.hop_ms) + 16
        self.encoder = build_audio_encoder(model_cfg.encoder, audio_cfg, max_len=max_audio_frames)
        cond_spec = default_condition_spec(tokenizer_cfg, cond_dim=model_cfg.decoder.d_model)
        self.cond_encoder = ConditionEncoder(cond_spec)
        self.decoder = TransformerDecoder(
            model_cfg.decoder,
            vocab_size_in=vocab_size_in,
            max_len=max_decoder_len,
            cond_dim=cond_spec.cond_dim,
        )
        encoder_dim = int(getattr(self.encoder, "output_dim", model_cfg.encoder.d_model))
        if encoder_dim != model_cfg.decoder.d_model:
            self.enc_to_dec = nn.Linear(encoder_dim, model_cfg.decoder.d_model)
        else:
            self.enc_to_dec = nn.Identity()
        self.vocab_size_out = vocab_size_out
        self.aux_heads = AuxHeads(
            encoder_dim=model_cfg.decoder.d_model,
            descriptor_count=cond_spec.descriptor_count,
        )
        self._descriptor_count = cond_spec.descriptor_count
        self.outliner: SongOutliner | None = None
        self._outliner_enabled = bool(model_cfg.outliner.enabled)
        if self._outliner_enabled:
            outliner_cfg = SongOutlinerConfig(
                enabled=True,
                mel_bins=audio_cfg.n_mels,
                summary_frames=model_cfg.outliner.summary_frames,
                num_anchors=model_cfg.outliner.num_anchors,
                d_model=model_cfg.outliner.d_model,
                out_dim=model_cfg.decoder.d_model,
                num_heads=model_cfg.outliner.num_heads,
                num_layers=model_cfg.outliner.num_layers,
                ffn_dim=model_cfg.outliner.ffn_dim,
                dropout=model_cfg.outliner.dropout,
            )
            self.outliner = SongOutliner(outliner_cfg)

    @property
    def cond_dim(self) -> int:
        return self.cond_encoder.cond_dim

    @property
    def descriptor_count(self) -> int:
        return self._descriptor_count

    def forward(
        self,
        mel: Tensor,
        input_ids: Tensor,
        cond_features: ConditionFeatures,
        summary_mel: Tensor | None = None,
        cond_null_mask: Tensor | None = None,
        mel_key_padding_mask: Tensor | None = None,
        token_key_padding_mask: Tensor | None = None,
    ) -> OsuformerOutput:
        memory = self.encode(mel, summary_mel=summary_mel, mel_key_padding_mask=mel_key_padding_mask)
        cond = self.cond_encoder(cond_features, null_mask=cond_null_mask)
        logits = self.decode(
            input_ids,
            memory=memory,
            cond=cond,
            token_key_padding_mask=token_key_padding_mask,
            memory_key_padding_mask=None,
        )
        pool = pool_encoder_output(memory, None)
        aux = self.aux_heads(pool)
        return OsuformerOutput(logits=logits, encoder_out=memory, aux=aux)

    def encode(
        self,
        mel: Tensor,
        summary_mel: Tensor | None = None,
        mel_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        window = self.encoder(mel, key_padding_mask=mel_key_padding_mask)
        memory = self.enc_to_dec(window)
        if self.outliner is not None and summary_mel is not None:
            anchors = self.outliner(summary_mel)
            memory = torch.cat([anchors, memory], dim=1)
        return memory

    def encode_condition(self, features: ConditionFeatures, null_mask: Tensor | None = None) -> Tensor:
        return self.cond_encoder(features, null_mask=null_mask)

    def null_condition(self, batch_size: int, device: torch.device) -> Tensor:
        return self.cond_encoder.null_vector(batch_size, device)

    def decode(
        self,
        input_ids: Tensor,
        memory: Tensor,
        cond: Tensor,
        token_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        dec = self.decoder(
            input_ids,
            memory=memory,
            cond=cond,
            tgt_key_padding_mask=token_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return F.linear(dec, self._tied_head_weight())

    def decode_step(
        self,
        input_ids: Tensor,
        memory: Tensor,
        cond: Tensor,
        cache: list[BlockCache] | None,
        start_pos: int,
        memory_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[BlockCache]]:
        dec, new_cache = self.decoder.step(
            input_ids,
            memory=memory,
            cond=cond,
            cache=cache,
            start_pos=start_pos,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return F.linear(dec, self._tied_head_weight()), new_cache

    def _tied_head_weight(self) -> Tensor:
        return self.decoder.embed.weight[: self.vocab_size_out]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
