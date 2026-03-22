from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

from .conformer import ConformerEncoder
from .transformer import TransformerDecoder


class Seq2SeqModel(nn.Module):
  def __init__(
    self,
    encoder: ConformerEncoder,
    decoder: TransformerDecoder,
  ) -> None:
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(
    self,
    *,
    src: torch.Tensor,
    tgt_tokens: torch.Tensor,
    src_key_padding_mask: Optional[torch.Tensor] = None,
    tgt_key_padding_mask: Optional[torch.Tensor] = None,
    conditioning: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    memory, memory_key_padding_mask = self.encoder(
      src,
      src_key_padding_mask,
      conditioning=conditioning,
    )

    logits = self.decoder(
      tgt_tokens,
      memory,
      tgt_key_padding_mask=tgt_key_padding_mask,
      memory_key_padding_mask=memory_key_padding_mask,
    )

    return logits

  @torch.no_grad()
  def greedy_decode(
    self,
    *,
    src: torch.Tensor,
    src_key_padding_mask: Optional[torch.Tensor] = None,
    bos_id: int = 1,
    eos_id: int = 2,
    max_len: int = 512,
  ) -> torch.Tensor:
    memory, memory_key_padding_mask = self.encoder(
      src,
      src_key_padding_mask,
    )

    return self.decoder.greedy_decode(
      memory,
      memory_key_padding_mask=memory_key_padding_mask,
      bos_id=bos_id,
      eos_id=eos_id,
      max_len=max_len,
    )