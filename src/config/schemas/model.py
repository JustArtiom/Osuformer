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
class OutlinerConfig:
    enabled: bool = True
    summary_frames: int = 600
    num_anchors: int = 32
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 3
    ffn_dim: int = 1024
    dropout: float = 0.1


@dataclass
class ModelConfig:
    version: str
    encoder: EncoderConfig
    decoder: DecoderConfig
    outliner: OutlinerConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.outliner is None:
            self.outliner = OutlinerConfig()
