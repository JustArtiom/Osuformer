from typing import TypedDict


class AudioConfig(TypedDict):
    version: str
    sample_rate: int
    hop_ms: int
    win_ms: int
    n_mels: int
    n_fft: int
    context_ms: int
    generate_ms: int
    lookahead_ms: int
