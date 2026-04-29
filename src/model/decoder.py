from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.config.schemas.model import DecoderConfig

from .adaln import AdaLNModulation, gated_residual, modulate
from .attention import MultiHeadAttention
from .positional import SinusoidalPositionalEncoding


@dataclass
class BlockCache:
    self_kv: tuple[Tensor, Tensor] | None = None
    cross_kv: tuple[Tensor, Tensor] | None = None


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float, cond_dim: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.adaln = AdaLNModulation(d_model=d_model, cond_dim=cond_dim, num_outputs=6)

    def _modulation(self, cond: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        outs = self.adaln(cond)
        scale_a, shift_a, gate_a, scale_f, shift_f, gate_f = outs
        return scale_a, shift_a, gate_a, scale_f, shift_f, gate_f

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        cond: Tensor,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        scale_a, shift_a, gate_a, scale_f, shift_f, gate_f = self._modulation(cond)

        residual = x
        x_norm = modulate(self.self_norm(x), scale_a, shift_a)
        attn_out = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=True,
        )
        x = gated_residual(residual, self.dropout(attn_out), gate_a)

        residual = x
        x_norm = self.cross_norm(x)
        cross_out = self.cross_attn(
            x_norm,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            is_causal=False,
        )
        x = residual + self.dropout(cross_out)

        residual = x
        x_norm = modulate(self.ffn_norm(x), scale_f, shift_f)
        x = gated_residual(residual, self.dropout(self.ffn(x_norm)), gate_f)
        return x

    def step(
        self,
        x: Tensor,
        memory: Tensor,
        cond: Tensor,
        cache: BlockCache,
        memory_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, BlockCache]:
        scale_a, shift_a, gate_a, scale_f, shift_f, gate_f = self._modulation(cond)

        residual = x
        x_norm = modulate(self.self_norm(x), scale_a, shift_a)
        q = self.self_attn.project_q(x_norm)
        new_k, new_v = self.self_attn.project_kv(x_norm, x_norm)
        if cache.self_kv is not None:
            past_k, past_v = cache.self_kv
            k = torch.cat([past_k, new_k], dim=2)
            v = torch.cat([past_v, new_v], dim=2)
            attn_out = self.self_attn.attend(q, k, v, key_padding_mask=None, is_causal=False)
        else:
            k, v = new_k, new_v
            attn_out = self.self_attn.attend(q, k, v, key_padding_mask=None, is_causal=True)
        x = gated_residual(residual, self.dropout(attn_out), gate_a)

        residual = x
        x_norm = self.cross_norm(x)
        q_cross = self.cross_attn.project_q(x_norm)
        if cache.cross_kv is not None:
            ck, cv = cache.cross_kv
        else:
            ck, cv = self.cross_attn.project_kv(memory, memory)
        cross_out = self.cross_attn.attend(
            q_cross, ck, cv, key_padding_mask=memory_key_padding_mask, is_causal=False
        )
        x = residual + self.dropout(cross_out)

        residual = x
        x_norm = modulate(self.ffn_norm(x), scale_f, shift_f)
        x = gated_residual(residual, self.dropout(self.ffn(x_norm)), gate_f)
        return x, BlockCache(self_kv=(k, v), cross_kv=(ck, cv))


class TransformerDecoder(nn.Module):
    def __init__(self, config: DecoderConfig, vocab_size_in: int, max_len: int, cond_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size_in, config.d_model)
        self.pos = SinusoidalPositionalEncoding(config.d_model, max_len=max_len)
        self.cond_dim = cond_dim
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                    cond_dim=cond_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor,
        cond: Tensor,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embed(input_ids)
        x = self.pos(x)
        for block in self.blocks:
            assert isinstance(block, TransformerDecoderBlock)
            x = block(
                x,
                memory=memory,
                cond=cond,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.final_norm(x)

    def step(
        self,
        input_ids: Tensor,
        memory: Tensor,
        cond: Tensor,
        cache: list[BlockCache] | None,
        start_pos: int,
        memory_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[BlockCache]]:
        x = self.embed(input_ids)
        x = self.pos(x, start_pos=start_pos)
        new_cache: list[BlockCache] = []
        for i, block in enumerate(self.blocks):
            assert isinstance(block, TransformerDecoderBlock)
            block_cache = cache[i] if cache is not None else BlockCache()
            x, updated = block.step(
                x,
                memory=memory,
                cond=cond,
                cache=block_cache,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            new_cache.append(updated)
        return self.final_norm(x), new_cache
