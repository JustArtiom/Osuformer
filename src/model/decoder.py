from __future__ import annotations

import torch
from torch import Tensor, nn

from src.config.schemas.model import DecoderConfig

from .positional import SinusoidalPositionalEncoding


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        residual = x
        x_norm = self.self_norm(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.cross_norm(x)
        cross_out, _ = self.cross_attn(
            x_norm, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(cross_out)

        residual = x
        x_norm = self.ffn_norm(x)
        x = residual + self.dropout(self.ffn(x_norm))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config: DecoderConfig, vocab_size_in: int, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size_in, config.d_model)
        self.pos = SinusoidalPositionalEncoding(config.d_model, max_len=max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embed(input_ids)
        x = self.pos(x)
        seq_len = x.size(1)
        causal_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=x.device)
        causal_mask = torch.triu(torch.ones_like(causal_mask), diagonal=1)
        for block in self.blocks:
            x = block(
                x,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.final_norm(x)
