from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn


def _causal_mask(t: int, device: torch.device) -> torch.Tensor:
  mask = torch.triu(torch.ones((t, t), device=device, dtype=torch.bool), diagonal=1)
  return mask.float().masked_fill(mask, float("-inf"))


class SinusoidalPositionalEncoding(nn.Module):
  pe: torch.Tensor
  def __init__(self, d_model: int, max_len: int = 32768):
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    t = x.size(1)
    return x + self.pe[:, :t, :].to(dtype=x.dtype)

class TransformerDecoderLayer(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
  ):
    super().__init__()

    self.self_attn = nn.MultiheadAttention(
      embed_dim=d_model,
      num_heads=n_heads,
      dropout=attn_dropout,
      batch_first=True,
    )
    self.cross_attn = nn.MultiheadAttention(
      embed_dim=d_model,
      num_heads=n_heads,
      dropout=attn_dropout,
      batch_first=True,
    )

    self.ff = nn.Sequential(
      nn.Linear(d_model, d_ff),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(d_ff, d_model),
    )

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)

    self.drop = nn.Dropout(dropout)

  def forward(
    self,
    x: torch.Tensor,
    memory: torch.Tensor,
    *,
    tgt_mask: Optional[torch.Tensor] = None,
    tgt_key_padding_mask: Optional[torch.Tensor] = None,
    memory_key_padding_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    q = self.norm1(x)
    attn_out, _ = self.self_attn(
      query=q,
      key=q,
      value=q,
      attn_mask=tgt_mask,
      key_padding_mask=tgt_key_padding_mask,
      need_weights=False,
    )
    x = x + self.drop(attn_out)

    q = self.norm2(x)
    kv = memory
    attn_out, _ = self.cross_attn(
      query=q,
      key=kv,
      value=kv,
      key_padding_mask=memory_key_padding_mask,
      need_weights=False,
    )
    x = x + self.drop(attn_out)

    q = self.norm3(x)
    ff_out = self.ff(q)
    x = x + self.drop(ff_out)

    return x


@dataclass
class TransformerDecoderConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 6
    dropout: float = 0.1
    attn_dropout: float = 0.0
    max_len: int = 32768
    pad_id: int = 0


class TransformerDecoder(nn.Module):
  def __init__(
    self, 
    vocab_size: int,
    d_model: int = 512,
    n_heads: int = 8,
    d_ff: int = 2048,
    n_layers: int = 6,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
    max_len: int = 32768,
    pad_id: int = 0,
  ):
    super().__init__()
    self.cfg = TransformerDecoderConfig(
      vocab_size=vocab_size,
      d_model=d_model,
      n_heads=n_heads,
      d_ff=d_ff,
      n_layers=n_layers,
      dropout=dropout,
      attn_dropout=attn_dropout,
      max_len=max_len,
      pad_id=pad_id,
    )

    self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model, padding_idx=self.cfg.pad_id)
    self.pos = SinusoidalPositionalEncoding(self.cfg.d_model, max_len=self.cfg.max_len)
    self.in_drop = nn.Dropout(self.cfg.dropout)

    self.layers = nn.ModuleList(
      [
        TransformerDecoderLayer(
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            d_ff=self.cfg.d_ff,
            dropout=self.cfg.dropout,
            attn_dropout=self.cfg.attn_dropout,
        )
        for _ in range(self.cfg.n_layers)
      ]
    )
    self.norm = nn.LayerNorm(self.cfg.d_model)
    self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

  def forward(
    self,
    tgt_tokens: torch.Tensor,
    memory: torch.Tensor,
    *,
    tgt_key_padding_mask: Optional[torch.Tensor] = None,
    memory_key_padding_mask: Optional[torch.Tensor] = None,
    use_causal_mask: bool = True,
  ) -> torch.Tensor:
    if tgt_tokens.dtype != torch.long:
      tgt_tokens = tgt_tokens.long()

    x = self.embed(tgt_tokens) * math.sqrt(self.cfg.d_model)
    x = self.pos(x)
    x = self.in_drop(x)

    tgt_mask = None
    if use_causal_mask:
      tgt_mask = _causal_mask(x.size(1), x.device)

    for layer in self.layers:
      x = layer(
          x,
          memory,
          tgt_mask=tgt_mask,
          tgt_key_padding_mask=tgt_key_padding_mask,
          memory_key_padding_mask=memory_key_padding_mask,
      )

    x = self.norm(x)
    logits = self.lm_head(x)
    return logits

  @torch.no_grad()
  def greedy_decode(
    self,
    memory: torch.Tensor,
    *,
    memory_key_padding_mask: Optional[torch.Tensor] = None,
    bos_id: int = 1,
    eos_id: int = 2,
    max_len: int = 512,
  ) -> torch.Tensor:
    """Simple greedy decoding (for debugging / sanity checks)."""
    b = memory.size(0)
    device = memory.device
    tokens = torch.full((b, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
      logits = self.forward(
        tokens,
        memory,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=memory_key_padding_mask,
        use_causal_mask=True,
      )
      next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
      tokens = torch.cat([tokens, next_token], dim=1)
      if (next_token.squeeze(1) == eos_id).all():
        break

    return tokens