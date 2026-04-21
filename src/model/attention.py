from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        b, lq, _ = query.shape
        lk = key.shape[1]
        q = self.q_proj(query).view(b, lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, lk, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout_p if self.training else 0.0

        if key_padding_mask is None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        else:
            pad = key_padding_mask.view(b, 1, 1, lk)
            attn_mask = torch.zeros(b, 1, 1, lk, device=q.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(pad, float("-inf"))
            if is_causal:
                causal = torch.triu(
                    torch.full((lq, lk), float("-inf"), device=q.device, dtype=q.dtype),
                    diagonal=1,
                )
                attn_mask = attn_mask + causal.view(1, 1, lq, lk)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(b, lq, self.d_model)
        return self.out_proj(out)
