from dataclasses import dataclass


@dataclass
class EncoderConfig:
    d_model: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    conv_kernel: int
    dropout: float


@dataclass
class DecoderConfig:
    d_model: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    dropout: float


@dataclass
class ModelConfig:
    version: str
    encoder: EncoderConfig
    decoder: DecoderConfig
