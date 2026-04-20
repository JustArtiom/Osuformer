from __future__ import annotations

import torch
from torch import Tensor, nn

from src.config.schemas.model import EncoderConfig

from .positional import SinusoidalPositionalEncoding


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel: int, dropout: float):
        super().__init__()
        assert kernel % 2 == 1
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = self.pw1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.dropout(x).transpose(1, 2)
        return residual + x


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.ffn1 = FeedForward(d_model, ffn_dim, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = FeedForward(d_model, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + 0.5 * self.ffn1(x)
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        x = self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, config: EncoderConfig, n_mels: int, max_len: int):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, config.d_model)
        self.pos = SinusoidalPositionalEncoding(config.d_model, max_len=max_len)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    conv_kernel=config.conv_kernel,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, mel: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = self.input_proj(mel)
        x = self.pos(x)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x
