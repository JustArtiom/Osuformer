# Vendored from https://github.com/minzwon/musicfm
# MIT License - Copyright 2023 ByteDance Inc. (see LICENSE in this directory)
#
# Adapted: imports made relative; type annotations added; training-only paths
# (masking, get_loss, get_targets, full forward) removed since we use this only
# as a pretrained feature extractor that we fine-tune.
from __future__ import annotations

import json
import random
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn

from .musicfm_conv import Conv2dSubsampling
from .musicfm_features import MelSTFT
from .musicfm_quantizer import RandomProjectionQuantizer


class MusicFM25Hz(nn.Module):
    def __init__(
        self,
        num_codebooks: int = 1,
        codebook_dim: int = 16,
        codebook_size: int = 4096,
        features: list[str] | None = None,
        hop_length: int = 240,
        n_mels: int = 128,
        conv_dim: int = 512,
        encoder_dim: int = 1024,
        encoder_depth: int = 12,
        mask_hop: float = 0.4,
        mask_prob: float = 0.6,
        is_flash: bool = False,
        stat_path: str | None = None,
        model_path: str | None = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = ["melspec_2048"]
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.features = features

        self.stat: dict[str, Any] = {}
        if stat_path is not None:
            with open(stat_path, "r") as f:
                self.stat = json.load(f)

        self.preprocessor_melspec_2048 = MelSTFT(n_fft=2048, hop_length=hop_length, is_db=True)

        seed = 142
        for feature in self.features:
            for i in range(num_codebooks):
                setattr(
                    self,
                    f"quantizer_{feature}_{i}",
                    RandomProjectionQuantizer(n_mels * 4, codebook_dim, codebook_size, seed=seed + i),
                )

        self.conv = Conv2dSubsampling(1, conv_dim, encoder_dim, strides=[2, 2], n_bands=n_mels)

        if is_flash:
            from .flash_conformer import (  # type: ignore[import-not-found]
                Wav2Vec2ConformerConfig,
                Wav2Vec2ConformerEncoder,
            )
        else:
            from transformers.models.wav2vec2_conformer.configuration_wav2vec2_conformer import (
                Wav2Vec2ConformerConfig,
            )
            from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
                Wav2Vec2ConformerEncoder,
            )
        config = Wav2Vec2ConformerConfig.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
        config.num_hidden_layers = encoder_depth
        config.hidden_size = encoder_dim
        self.conformer = Wav2Vec2ConformerEncoder(config)

        self.linear = nn.Linear(encoder_dim, codebook_size)

        random.seed(seed)
        self.cls_token = nn.Parameter(torch.randn(encoder_dim))

        if model_path:
            state = torch.load(model_path, map_location="cpu")
            sd = state["state_dict"] if "state_dict" in state else state
            sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
            sd = {k: v for k, v in sd.items() if not k.startswith("preprocessor_melspec_2048.")}
            self.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def preprocessing(self, x: Tensor, features: list[str]) -> dict[str, Tensor]:
        precision = 16 if x.dtype == torch.float16 else 32
        out: dict[str, Tensor] = {}
        for key in features:
            layer = getattr(self, f"preprocessor_{key}")
            out[key] = layer.float()(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
        return out

    @torch.no_grad()
    def normalize(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        for key in x.keys():
            mean = self.stat[f"{key}_mean"]
            std = self.stat[f"{key}_std"]
            x[key] = (x[key] - mean) / std
        return x

    @torch.no_grad()
    def rearrange(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        for key in x.keys():
            if key == "chromagram":
                x[key] = rearrange(x[key], "b f t -> b t f")
            else:
                x[key] = rearrange(x[key], "b f (t s) -> b t (s f)", s=4)
        return x

    def encoder(self, x: Tensor) -> tuple[dict[str, Tensor], tuple[Tensor, ...]]:
        x = self.conv(x)
        out = self.conformer(x, output_hidden_states=True)
        hidden_emb: tuple[Tensor, ...] = out["hidden_states"]
        last_emb: Tensor = out["last_hidden_state"]
        proj: Tensor = self.linear(last_emb)
        logits = {
            key: proj[:, :, i * self.codebook_size : (i + 1) * self.codebook_size]
            for i, key in enumerate(self.features)
        }
        return logits, hidden_emb

    def get_predictions(self, x: Tensor) -> tuple[dict[str, Tensor], tuple[Tensor, ...]]:
        feats = self.preprocessing(x, features=["melspec_2048"])
        feats = self.normalize(feats)
        return self.encoder(feats["melspec_2048"])

    def get_latent(self, x: Tensor, layer_ix: int = 12) -> Tensor:
        _, hidden_states = self.get_predictions(x)
        return hidden_states[layer_ix]
