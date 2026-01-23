from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class FeedForwardModule(nn.Module):
  def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
    super().__init__()
    self.linear1 = nn.Linear(d_model, ffn_dim)
    self.activation = Swish()
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(ffn_dim, d_model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear2(self.dropout(self.activation(self.linear1(x))))


class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000) -> None:
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.d_model = d_model
    self.register_buffer("pe", self._build_pe(max_len), persistent=False)

  def _build_pe(self, max_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    position = torch.arange(0, max_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
    pe = torch.zeros(1, max_len, self.d_model, device=device)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe

  def _maybe_extend_pe(self, length: int, device: torch.device) -> None:
    if length <= self.pe.size(1):
        return
    new_len = max(length, self.pe.size(1) * 2)
    self.pe = self._build_pe(new_len, device=device)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    length = x.size(1)
    self._maybe_extend_pe(length, x.device)
    x = x + self.pe[:, :length]
    return self.dropout(x)
  

class MultiHeadSelfAttention(nn.Module):
  def __init__(
    self,
    d_model: int,
    num_heads: int,
    dropout: float,
    use_relative_attention: bool = False,
    max_relative_position: int = 128,
    relative_style: str = "bias",
  ) -> None:
    super().__init__()
    self.use_relative_attention = use_relative_attention
    self.num_heads = num_heads
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)
    self.relative_style = (relative_style or "bias").lower()
    if not use_relative_attention:
      self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
    else:
      if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads for relative attention.")
      self.head_dim = d_model // num_heads
      self.q_proj = nn.Linear(d_model, d_model)
      self.k_proj = nn.Linear(d_model, d_model)
      self.v_proj = nn.Linear(d_model, d_model)
      self.out_proj = nn.Linear(d_model, d_model)
      self.max_rel_pos = max(1, int(max_relative_position))
      if self.relative_style == "bias":
        self.rel_bias = nn.Embedding(2 * self.max_rel_pos + 1, num_heads)
        self.u = None
        self.v = None
      else:
        self.rel_bias = None
        self.u = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.v = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

  def forward(
    self,
    x: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    if not self.use_relative_attention:
      out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
      return self.dropout(out)

    bsz, seq_len, _ = x.shape
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

    if self.relative_style == "bias":
      positions = torch.arange(seq_len, device=x.device)
      rel_positions = positions[None, :] - positions[:, None]
      rel_positions = rel_positions.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
      assert self.rel_bias is not None
      rel_bias = self.rel_bias(rel_positions)
      rel_bias = rel_bias.permute(2, 0, 1)
      scores = scores + rel_bias.unsqueeze(0)
    else:
      positions = torch.arange(seq_len, device=x.device)
      rel_positions = positions[None, :] - positions[:, None]
      rel_positions = rel_positions.clamp(-self.max_rel_pos, self.max_rel_pos)
      offset = self.max_rel_pos
      rel_range = torch.arange(-self.max_rel_pos, self.max_rel_pos + 1, device=x.device)
      inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim))
      sinusoid_inp = torch.einsum("i,j->ij", rel_range.float(), inv_freq)
      rel_emb = torch.zeros((rel_range.numel(), self.head_dim), device=x.device)
      rel_emb[:, 0::2] = torch.sin(sinusoid_inp)
      rel_emb[:, 1::2] = torch.cos(sinusoid_inp)
      rel_index = (rel_positions + offset).long()
      rel_emb_mat = rel_emb[rel_index]

      assert self.u is not None
      assert self.v is not None
      q_with_u = q + self.u.unsqueeze(0).unsqueeze(2)
      q_with_v = q + self.v.unsqueeze(0).unsqueeze(2)

      content_scores = torch.matmul(q_with_u, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
      rel_scores = torch.einsum("bhid,ijd->bhij", q_with_v, rel_emb_mat) / math.sqrt(self.head_dim)
      scores = content_scores + rel_scores

    if attn_mask is not None:
      if attn_mask.dtype == torch.bool:
        scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
      else:
        scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

    if key_padding_mask is not None:
      mask = key_padding_mask[:, None, None, :].to(torch.bool)
      scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = self.dropout(attn)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
    out = self.out_proj(out)
    return self.dropout(out)


class ConvolutionModule(nn.Module):
  def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
    super().__init__()
    padding = (kernel_size - 1) // 2
    self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
    self.glu = nn.GLU(dim=1)
    self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model)
    self.batch_norm = nn.BatchNorm1d(d_model)
    self.activation = Swish()
    self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)
    x = self.pointwise_conv1(x)
    x = self.glu(x)
    x = self.depthwise_conv(x)
    x = self.batch_norm(x)
    x = self.activation(x)
    x = self.pointwise_conv2(x)
    x = self.dropout(x)
    return x.transpose(1, 2)


class ConformerBlock(nn.Module):
  def __init__(
    self,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    conv_kernel: int,
    dropout: float,
    use_relative_attention: bool = False,
    max_relative_position: int = 128,
    relative_style: str = "bias",
  ) -> None:
    super().__init__()
    self.ffn1 = FeedForwardModule(d_model, ffn_dim, dropout)
    self.self_attn = MultiHeadSelfAttention(
      d_model,
      num_heads,
      dropout,
      use_relative_attention=use_relative_attention,
      max_relative_position=max_relative_position,
      relative_style=relative_style,
    )
    self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
    self.ffn2 = FeedForwardModule(d_model, ffn_dim, dropout)

    self.norm_ffn1 = nn.LayerNorm(d_model)
    self.norm_attn = nn.LayerNorm(d_model)
    self.norm_conv = nn.LayerNorm(d_model)
    self.norm_ffn2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    residual = x
    x = residual + 0.5 * self.dropout(self.ffn1(self.norm_ffn1(x)))
    residual = x
    x = residual + self.self_attn(self.norm_attn(x), key_padding_mask=key_padding_mask)
    residual = x
    x = residual + self.conv(self.norm_conv(x))
    residual = x
    x = residual + 0.5 * self.dropout(self.ffn2(self.norm_ffn2(x)))
    return x


class Conv2dSubsampling(nn.Module):
  def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
    )
    out_freq = self._output_length(input_dim)
    out_freq = self._output_length(out_freq)
    self.linear = nn.Linear(d_model * out_freq, d_model)
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def _output_length(length: int) -> int:
    return (length + 1) // 2

  def _subsample_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
      return None
    valid = (~mask).unsqueeze(1).float()
    valid = F.max_pool1d(valid, kernel_size=3, stride=2, padding=1)
    valid = F.max_pool1d(valid, kernel_size=3, stride=2, padding=1)
    new_valid = valid.squeeze(1) > 0.0
    return ~new_valid

  def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    bsz, time, feat = x.shape
    x = x.view(bsz, 1, time, feat)
    x = self.conv(x)
    bsz, channels, t_sub, f_sub = x.shape
    x = x.transpose(1, 2).contiguous().view(bsz, t_sub, channels * f_sub)
    x = self.linear(x)
    x = self.dropout(x)
    new_mask = self._subsample_mask(mask)
    return x, new_mask


class ConformerEncoder(nn.Module):
  def __init__(
    self,
    input_dim: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    ffn_dim: int,
    conv_kernel: int,
    dropout: float,
    positional_encoding: str = "absolute",
    max_relative_position: int = 128,
    relative_style: str = "bias",
    subsampling: bool = False,
  ) -> None:
    super().__init__()
    self.use_absolute_positional_encoding = positional_encoding.lower() != "relative"
    proj_in_dim = d_model if subsampling else input_dim
    self.input_proj = nn.Linear(proj_in_dim, d_model)
    self.positional_encoding = PositionalEncoding(d_model, dropout) if self.use_absolute_positional_encoding else None
    self.subsampling = Conv2dSubsampling(input_dim, d_model, dropout) if subsampling else None
    self.layers = nn.ModuleList(
        [
            ConformerBlock(
                d_model,
                num_heads,
                ffn_dim,
                conv_kernel,
                dropout,
                use_relative_attention=not self.use_absolute_positional_encoding,
                max_relative_position=max_relative_position,
                relative_style=relative_style,
            )
            for _ in range(num_layers)
        ]
    )
    self.norm = nn.LayerNorm(d_model)

  def forward(
      self,
      x: torch.Tensor,
      src_key_padding_mask: Optional[torch.Tensor],
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if self.subsampling is not None:
      x, new_mask = self.subsampling(x, src_key_padding_mask)
      if new_mask is not None:
        all_masked = new_mask.all(dim=1)
        if all_masked.any():
          new_mask = new_mask.clone()
          new_mask[all_masked, 0] = False
      src_key_padding_mask = new_mask
    x = self.input_proj(x)
    if self.positional_encoding is not None:
      x = self.positional_encoding(x)
    for layer in self.layers:
      x = layer(x, key_padding_mask=src_key_padding_mask)
    return self.norm(x), src_key_padding_mask
