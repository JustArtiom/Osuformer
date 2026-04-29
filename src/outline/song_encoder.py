from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.model.attention import MultiHeadAttention
from src.model.positional import SinusoidalPositionalEncoding


@dataclass
class SongOutlinerConfig:
    enabled: bool
    mel_bins: int
    summary_frames: int
    num_anchors: int
    d_model: int
    out_dim: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    dropout: float


class _OutlinerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, x_norm, x_norm, key_padding_mask=None, is_causal=False)
        x = residual + self.dropout(attn_out)
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.dropout(self.ffn(x_norm))
        return x


class SongOutliner(nn.Module):
    def __init__(self, config: SongOutlinerConfig):
        super().__init__()
        self.config = config
        self.proj_in = nn.Linear(config.mel_bins, config.d_model)
        self.pos = SinusoidalPositionalEncoding(config.d_model, max_len=config.summary_frames)
        self.blocks = nn.ModuleList(
            [
                _OutlinerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.queries = nn.Parameter(torch.randn(config.num_anchors, config.d_model) * 0.02)
        self.attn_pool = MultiHeadAttention(config.d_model, config.num_heads, dropout=config.dropout)
        self.proj_out = nn.Linear(config.d_model, config.out_dim)

    @property
    def num_anchors(self) -> int:
        return self.config.num_anchors

    @property
    def out_dim(self) -> int:
        return self.config.out_dim

    def forward(self, summary_mel: Tensor) -> Tensor:
        x = self.proj_in(summary_mel)
        x = self.pos(x)
        for block in self.blocks:
            assert isinstance(block, _OutlinerBlock)
            x = block(x)
        x = self.norm(x)
        b = x.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1).contiguous()
        anchors = self.attn_pool(q, x, x, key_padding_mask=None, is_causal=False)
        return self.proj_out(anchors)


def downsample_mel_to_summary(mel: Tensor, summary_frames: int) -> Tensor:
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    n = mel.shape[1]
    if n == 0:
        return torch.zeros(mel.shape[0], summary_frames, mel.shape[2], dtype=mel.dtype, device=mel.device)
    if n < summary_frames:
        pad = torch.zeros(mel.shape[0], summary_frames - n, mel.shape[2], dtype=mel.dtype, device=mel.device)
        mel = torch.cat([mel, pad], dim=1)
    x = mel.transpose(1, 2)
    pooled = torch.nn.functional.adaptive_avg_pool1d(x, output_size=summary_frames)
    return pooled.transpose(1, 2)
