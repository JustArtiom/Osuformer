from dataclasses import dataclass


@dataclass
class AudioConfig:
    version: str
    sample_rate: int
    hop_ms: int
    win_ms: int
    n_mels: int
    n_fft: int
    context_ms: int
    generate_ms: int
    lookahead_ms: int
    preset: str = "default"
    stats_path: str | None = None
