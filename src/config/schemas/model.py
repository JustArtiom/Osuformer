from typing import TypedDict


class EncoderConfig(TypedDict):
    d_model: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    conv_kernel: int
    dropout: float


class DecoderConfig(TypedDict):
    d_model: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    dropout: float


class ModelConfig(TypedDict):
    version: str
    encoder: EncoderConfig
    decoder: DecoderConfig
