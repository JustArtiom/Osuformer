from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class KVCache:
    k: torch.Tensor        # (B, H, max_len, D)
    v: torch.Tensor        # (B, H, max_len, D)
    length: int            # current sequence length

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

  def forward_step(
    self,
    x: torch.Tensor,
    memory: torch.Tensor,
    *,
    cache: Optional[KVCache] = None,
    memory_key_padding_mask: Optional[torch.Tensor] = None,
    max_len: int,
  ) -> tuple[torch.Tensor, KVCache]:
    if x.size(1) != 1:
      raise ValueError("forward_step expects a single-step sequence (T=1).")

    q = self.norm1(x)
    attn_out, new_cache = self._self_attn_step(q, cache, max_len=max_len)
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
    return x, new_cache

  def _self_attn_step(
    self,
    x: torch.Tensor,
    cache: Optional[KVCache],
    *,
    max_len: int,
  ) -> tuple[torch.Tensor, KVCache]:
    mha = self.self_attn
    bsz, tgt_len, _ = x.shape
    if tgt_len != 1:
      raise ValueError("_self_attn_step expects tgt_len=1.")

    embed_dim = mha.embed_dim
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads

    # Project QKV
    if mha.in_proj_weight is not None:
      qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
      q, k, v = qkv.chunk(3, dim=-1)
    else:
      if mha.in_proj_bias is None:
        q_bias = k_bias = v_bias = None
      else:
        q_bias, k_bias, v_bias = mha.in_proj_bias.chunk(3)
      q = F.linear(x, mha.q_proj_weight, q_bias)
      k = F.linear(x, mha.k_proj_weight, k_bias)
      v = F.linear(x, mha.v_proj_weight, v_bias)

    # (B, 1, E) -> (B, H, 1, D)
    q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)

    # Lazy init preallocated cache
    if cache is None:
      if max_len <= 0:
        raise ValueError(f"max_len must be > 0, got {max_len}.")
      cache = KVCache(
        k=torch.empty((bsz, num_heads, max_len, head_dim), device=x.device, dtype=k.dtype),
        v=torch.empty((bsz, num_heads, max_len, head_dim), device=x.device, dtype=v.dtype),
        length=0,
      )

    if cache.length >= cache.k.size(2):
      raise ValueError(
        f"KV cache overflow: length={cache.length} >= max_len={cache.k.size(2)}. "
        "Increase max_len or reduce generation length."
      )

    # Write this step into the cache (in-place, no cat)
    t = cache.length
    cache.k[:, :, t : t + 1, :] = k
    cache.v[:, :, t : t + 1, :] = v
    cache.length += 1

    # Attention over cached keys/values
    k_full = cache.k[:, :, : cache.length, :]
    v_full = cache.v[:, :, : cache.length, :]

    attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_weights = F.dropout(attn_weights, p=mha.dropout, training=self.training)
    attn_out = torch.matmul(attn_weights, v_full)

    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
    attn_out = mha.out_proj(attn_out)

    return attn_out, cache


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
      layer_mod = cast(TransformerDecoderLayer, layer)
      x = layer_mod(
        x,
        memory,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
      )

    x = self.norm(x)
    logits = self.lm_head(x)
    return logits

  def init_kv_cache(
    self,
    *,
    bsz: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_len: int | None = None,
  ) -> list[KVCache]:
    """Allocate a pre-sized KV cache for fast incremental decoding."""
    if dtype is None:
      dtype = next(self.parameters()).dtype
    if max_len is None:
      max_len = self.cfg.max_len

    if bsz <= 0:
      raise ValueError(f"bsz must be > 0, got {bsz}.")
    if max_len <= 0:
      raise ValueError(f"max_len must be > 0, got {max_len}.")

    head_dim = self.cfg.d_model // self.cfg.n_heads
    caches: list[KVCache] = []
    for _ in range(len(self.layers)):
      caches.append(
        KVCache(
          k=torch.empty((bsz, self.cfg.n_heads, max_len, head_dim), device=device, dtype=dtype),
          v=torch.empty((bsz, self.cfg.n_heads, max_len, head_dim), device=device, dtype=dtype),
          length=0,
        )
      )
    return caches

  @torch.no_grad()
  def forward_step(
    self,
    token_ids: torch.Tensor,
    memory: torch.Tensor,
    *,
    cache: list[KVCache],
    memory_key_padding_mask: Optional[torch.Tensor] = None,
    position: int,
  ) -> tuple[torch.Tensor, list[KVCache]]:
    if token_ids.dtype != torch.long:
      token_ids = token_ids.long()
    if token_ids.dim() == 1:
      token_ids = token_ids.unsqueeze(1)

    x = self.embed(token_ids) * math.sqrt(self.cfg.d_model)
    if position < 0 or position >= self.pos.pe.size(1):
      raise ValueError(f"Position {position} out of range for positional encoding (max={self.pos.pe.size(1)-1}).")
    x = x + self.pos.pe[:, position : position + 1, :].to(dtype=x.dtype)
    x = self.in_drop(x)

    new_cache: list[KVCache] = []
    for layer, layer_cache in zip(self.layers, cache):
      layer_mod = cast(TransformerDecoderLayer, layer)
      out, updated = layer_mod.forward_step(
        x,
        memory,
        cache=layer_cache,
        memory_key_padding_mask=memory_key_padding_mask,
        max_len=cache[0].k.size(2) if cache else self.cfg.max_len,
      )
      x = out
      new_cache.append(updated)

    x = self.norm(x)
    logits = self.lm_head(x)
    return logits, new_cache

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
