from typing import TypedDict


class TokenizerConfig(TypedDict):
    version: str
    dt_bin_ms: int
    coordinate_step: int
    coordinate_padding: int
