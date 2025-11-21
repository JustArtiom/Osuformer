from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.tokenizer import HitObjectTokenizer, TokenAttr, TokenType


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

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B, H, T, T]

        if self.relative_style == "bias":
            # Learned bias per bucket
            positions = torch.arange(seq_len, device=x.device)
            rel_positions = positions[None, :] - positions[:, None]  # [T, T]
            rel_positions = rel_positions.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
            rel_bias = self.rel_bias(rel_positions)  # [T, T, H]
            rel_bias = rel_bias.permute(2, 0, 1)  # [H, T, T]
            scores = scores + rel_bias.unsqueeze(0)  # broadcast over batch
        else:
            # Transformer-XL style relative attention with sinusoidal embeddings and u/v biases.
            positions = torch.arange(seq_len, device=x.device)
            rel_positions = positions[None, :] - positions[:, None]  # [T, T], range [-(T-1), T-1]
            rel_positions = rel_positions.clamp(-self.max_rel_pos, self.max_rel_pos)
            offset = self.max_rel_pos
            # Build sinusoidal embeddings for buckets [-max_rel_pos, max_rel_pos]
            rel_range = torch.arange(-self.max_rel_pos, self.max_rel_pos + 1, device=x.device)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim))
            sinusoid_inp = torch.einsum("i,j->ij", rel_range.float(), inv_freq)  # [2*max+1, d/2]
            rel_emb = torch.zeros((rel_range.numel(), self.head_dim), device=x.device)
            rel_emb[:, 0::2] = torch.sin(sinusoid_inp)
            rel_emb[:, 1::2] = torch.cos(sinusoid_inp)
            rel_index = (rel_positions + offset).long()  # [T, T]
            rel_emb_mat = rel_emb[rel_index]  # [T, T, D]

            q_with_u = q + self.u.unsqueeze(0).unsqueeze(2)  # [B, H, T, D]
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
            mask = key_padding_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, T]
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_proj(out)
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


class Conv2dSubsampling(nn.Module):
    """2D convolutional subsampling (stride 2x2 twice) followed by linear projection."""

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
        # Conv length with kernel=3, stride=2, padding=1
        return (length + 1) // 2

    def _subsample_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        # mask: [B, T] bool (True for padding)
        lengths = (~mask).sum(dim=1)
        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if max_len == 0:
            return torch.ones(mask.size(0), 0, dtype=torch.bool, device=mask.device)
        new_positions = torch.arange(max_len, device=mask.device).unsqueeze(0)
        new_mask = new_positions >= lengths.unsqueeze(1)
        return new_mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [B, T, F]
        bsz, time, feat = x.shape
        x = x.view(bsz, 1, time, feat)
        x = self.conv(x)  # [B, C, T', F']
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
        self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.subsampling is not None:
            x, src_key_padding_mask = self.subsampling(x, src_key_padding_mask)
        x = self.input_proj(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=src_key_padding_mask)
        return self.norm(x), src_key_padding_mask


class HitObjectDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        use_relative_self_attn: bool = False,
        relative_style: str = "bias",
        max_relative_position: int = 128,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            dropout,
            use_relative_attention=use_relative_self_attn,
            max_relative_position=max_relative_position,
            relative_style=relative_style,
        )
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardModule(d_model, ffn_dim, dropout)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.dropout1(self.self_attn(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask))

    def _ca_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        mem_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out, _ = self.cross_attn(x, mem, mem, key_padding_mask=mem_key_padding_mask, need_weights=False)
        return self.dropout2(out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout3(self.ffn(x))

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._ca_block(self.norm2(x), memory, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._ca_block(x, memory, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        return x


class HitObjectDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attr_sizes: Sequence[int],
        positional_encoding: str = "absolute",
        relative_style: str = "bias",
        max_relative_position: int = 128,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.attr_sizes = list(attr_sizes)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, d_model, padding_idx=0) for size in self.attr_sizes]
        )
        self.use_absolute_positional_encoding = positional_encoding.lower() != "relative"
        self.positional_encoding = PositionalEncoding(d_model, dropout) if self.use_absolute_positional_encoding else None
        self.layers = nn.ModuleList(
            [
                HitObjectDecoderLayer(
                    d_model,
                    num_heads,
                    ffn_dim,
                    dropout,
                    use_relative_self_attn=not self.use_absolute_positional_encoding,
                    relative_style=relative_style,
                    max_relative_position=max_relative_position,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model) if norm_first else None
        self.bos = nn.Parameter(torch.zeros(d_model))
        self.attr_heads = nn.ModuleList([nn.Linear(d_model, size) for size in self.attr_sizes])

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        embed = None
        for idx, embedding in enumerate(self.embeddings):
            attr_emb = embedding(tokens[..., idx])
            embed = attr_emb if embed is None else embed + attr_emb
        return embed

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        targets: torch.Tensor,
        target_key_padding_mask: Optional[torch.Tensor],
        temperature: float = 1.0,
    ) -> List[torch.Tensor]:
        batch_size, target_len, _ = targets.shape
        embedded = self._embed_tokens(targets)
        shifted = torch.zeros_like(embedded)
        if target_len > 1:
            shifted[:, 1:, :] = embedded[:, :-1, :]
        shifted[:, 0, :] = self.bos
        if self.positional_encoding is not None:
            shifted = self.positional_encoding(shifted)

        tgt_mask = self._causal_mask(target_len, memory.device)

        decoded = shifted
        for layer in self.layers:
            decoded = layer(
                decoded,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if self.final_norm is not None:
            decoded = self.final_norm(decoded)

        logits_list: List[torch.Tensor] = []
        for head in self.attr_heads:
            logits = head(decoded)
            if temperature != 1.0:
                temp = max(temperature, 1e-4)
                logits = logits / temp
            logits_list.append(logits)
        return logits_list


class ConformerSeq2Seq(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        encoder_cfg = config["model"]["encoder"]
        decoder_cfg = config["model"]["decoder"]
        data_cfg = config["data"]
        audio_cfg = config["audio"]

        input_dim = audio_cfg["n_mels"]
        pos_enc_type = encoder_cfg.get("positional_encoding", "absolute")
        max_rel_pos = int(encoder_cfg.get("relative_max_position", 128))
        relative_style = encoder_cfg.get("relative_style", "bias")
        subsampling_enabled = bool(encoder_cfg.get("subsampling", False))

        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=encoder_cfg["d_model"],
            num_layers=encoder_cfg["num_layers"],
            num_heads=encoder_cfg["num_heads"],
            ffn_dim=encoder_cfg["ffn_dim"],
            conv_kernel=encoder_cfg["conv_kernel"],
            dropout=encoder_cfg.get("dropout", 0.1),
            positional_encoding=pos_enc_type,
            max_relative_position=max_rel_pos,
            relative_style=relative_style,
            subsampling=subsampling_enabled,
        )

        self.data_cfg_snapshot = data_cfg.copy()
        self.tokenizer = HitObjectTokenizer(data_cfg)
        self.attr_sizes = self.tokenizer.attribute_sizes
        context_beats = data_cfg.get("context_beats", 8)
        target_beats = data_cfg.get("target_beats", 16)
        ticks_per_beat = data_cfg.get("ticks_per_beat", 4)
        self.seq_len = (context_beats + target_beats) * ticks_per_beat

        decoder_pos_enc = decoder_cfg.get("positional_encoding", "absolute")
        decoder_relative_style = decoder_cfg.get("relative_style", "bias")
        decoder_max_rel_pos = int(decoder_cfg.get("relative_max_position", 128))
        decoder_norm_first = bool(decoder_cfg.get("norm_first", False))

        self.decoder = HitObjectDecoder(
            d_model=encoder_cfg["d_model"],
            num_layers=decoder_cfg["num_layers"],
            num_heads=decoder_cfg["num_heads"],
            ffn_dim=decoder_cfg["ffn_dim"],
            dropout=decoder_cfg.get("dropout", 0.1),
            attr_sizes=self.attr_sizes,
            positional_encoding=decoder_pos_enc,
            relative_style=decoder_relative_style,
            max_relative_position=decoder_max_rel_pos,
            norm_first=decoder_norm_first,
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_mask: Optional[torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> List[torch.Tensor]:
        memory, enc_mask = self.encoder(audio, audio_mask)
        if targets is None:
            raise ValueError("Targets are required for training forward pass")
        attr_logits = self.decoder(memory, enc_mask, targets, target_mask, temperature=temperature)
        return attr_logits

    @torch.no_grad()
    def generate(
        self,
        audio: torch.Tensor,
        audio_mask: Optional[torch.Tensor],
        prompt: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        memory, enc_mask = self.encoder(audio, audio_mask)
        batch_size = audio.size(0)
        attr_count = TokenAttr.COUNT
        if prompt is not None:
            prompt = prompt.to(audio.device)
            generated = prompt.clone()
            prompt_len = prompt.size(1)
        else:
            generated = torch.zeros(batch_size, 0, attr_count, device=audio.device, dtype=torch.long)
            prompt_len = 0

        total_limit = self.seq_len
        remaining = total_limit - prompt_len
        if max_steps is not None:
            remaining = min(remaining, int(max_steps))
        remaining = max(0, remaining)

        outputs: List[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=audio.device)

        for _ in range(remaining):
            placeholder = torch.zeros(batch_size, 1, attr_count, device=audio.device, dtype=torch.long)
            decoder_input = torch.cat([generated, placeholder], dim=1)
            logits_list = self.decoder(memory, enc_mask, decoder_input, None, temperature=temperature)

            assembled = torch.zeros(batch_size, attr_count, device=audio.device, dtype=torch.long)
            for attr_idx, logits in enumerate(logits_list):
                attr_logits = logits[:, -1, :]
                probs = F.softmax(attr_logits, dim=-1)
                sample = torch.multinomial(probs, num_samples=1).squeeze(-1)
                assembled[:, attr_idx] = sample

            token_type = assembled[:, TokenAttr.TYPE]
            eos_mask = token_type == TokenType.EOS
            slider_mask = token_type == TokenType.SLIDER

            assembled[:, TokenAttr.TICK] = torch.clamp(
                assembled[:, TokenAttr.TICK], min=1, max=self.seq_len + 1
            )
            assembled[~slider_mask, TokenAttr.DURATION :] = 0
            if eos_mask.any():
                assembled[eos_mask, TokenAttr.TICK :] = 0

            outputs.append(assembled)
            generated = torch.cat([generated, assembled.unsqueeze(1)], dim=1)
            finished |= eos_mask
            if finished.all():
                break

        if not outputs:
            outputs.append(torch.zeros(batch_size, attr_count, device=audio.device, dtype=torch.long))
        return torch.stack(outputs, dim=1)

    def tokenizer_meta(self) -> dict:
        return {
            "attr_sizes": self.attr_sizes,
            "seq_len": self.seq_len,
            "data_cfg": self.data_cfg_snapshot,
        }
