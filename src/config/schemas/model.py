from dataclasses import dataclass


@dataclass
class EncoderConfig:
    d_model: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    conv_kernel: int
    dropout: float
    type: str = "conformer_scratch"
    musicfm_model_path: str | None = None
    musicfm_stats_path: str | None = None
    musicfm_layer: int = 12
    freeze_first_n_layers: int = 0


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
