from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout(out)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
        )
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
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, conv_kernel: int, dropout: float) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
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
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(d_model, num_heads, ffn_dim, conv_kernel, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=src_key_padding_mask)
        return self.norm(x)


class HitObjectDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.event_embed = nn.Linear(4, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.bos = nn.Parameter(torch.zeros(d_model))
        self.output_proj = nn.Linear(d_model, 4)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        targets: torch.Tensor,
        target_key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, target_len, _ = targets.shape
        embedded = self.event_embed(targets)
        shifted = torch.zeros_like(embedded)
        if target_len > 1:
            shifted[:, 1:, :] = embedded[:, :-1, :]
        shifted[:, 0, :] = self.bos
        shifted = self.positional_encoding(shifted)

        tgt_mask = self._causal_mask(target_len, memory.device)

        decoded = self.decoder(
            shifted,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.output_proj(decoded)
        coord_logits = logits[..., :3]
        eos_logits = logits[..., 3:]
        coord_pred = torch.sigmoid(coord_logits)
        return coord_pred, eos_logits


class ConformerSeq2Seq(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        encoder_cfg = config["model"]["encoder"]
        decoder_cfg = config["model"]["decoder"]
        data_cfg = config["data"]

        self.encoder = ConformerEncoder(
            input_dim=data_cfg["n_mels"],
            d_model=encoder_cfg["d_model"],
            num_layers=encoder_cfg["num_layers"],
            num_heads=encoder_cfg["num_heads"],
            ffn_dim=encoder_cfg["ffn_dim"],
            conv_kernel=encoder_cfg["conv_kernel"],
            dropout=encoder_cfg.get("dropout", 0.1),
        )

        self.decoder = HitObjectDecoder(
            d_model=encoder_cfg["d_model"],
            num_layers=decoder_cfg["num_layers"],
            num_heads=decoder_cfg["num_heads"],
            ffn_dim=decoder_cfg["ffn_dim"],
            dropout=decoder_cfg.get("dropout", 0.1),
        )

        self.max_seq_len = decoder_cfg.get("max_seq_len", 64)

    def forward(
        self,
        audio: torch.Tensor,
        audio_mask: Optional[torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = self.encoder(audio, audio_mask)
        if targets is None:
            raise ValueError("Targets are required for training forward pass")
        coords, eos_logits = self.decoder(memory, audio_mask, targets, target_mask)
        return coords, eos_logits

    @torch.no_grad()
    def generate(
        self,
        audio: torch.Tensor,
        audio_mask: Optional[torch.Tensor],
        max_steps: Optional[int] = None,
        eos_threshold: float = 0.6,
    ) -> torch.Tensor:
        self.eval()
        memory = self.encoder(audio, audio_mask)
        batch_size = audio.size(0)
        max_steps = max_steps or self.max_seq_len

        generated = torch.zeros(batch_size, 1, 4, device=audio.device, dtype=audio.dtype)
        outputs: List[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=audio.device)

        for _ in range(max_steps):
            coords, eos_logits = self.decoder(memory, audio_mask, generated, None)
            next_coords = coords[:, -1, :]
            next_eos = torch.sigmoid(eos_logits[:, -1, :])
            outputs.append(torch.cat([next_coords, next_eos], dim=-1))

            eos_hit = (next_eos.squeeze(-1) > eos_threshold)
            finished = finished | eos_hit
            if finished.all():
                break

            next_token = torch.zeros(batch_size, 1, 4, device=audio.device, dtype=audio.dtype)
            next_token[:, 0, :3] = next_coords
            next_token[:, 0, 3] = 0.0
            generated = torch.cat([generated, next_token], dim=1)

        return torch.stack(outputs, dim=1)
