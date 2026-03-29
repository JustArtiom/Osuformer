from dataclasses import dataclass


@dataclass
class PathsConfig:
    root: str
    data: str
    cache: str
    checkpoints: str
    logs: str
