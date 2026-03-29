from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    version: str
    dt_bin_ms: int
    coordinate_step: int
    coordinate_padding: int
