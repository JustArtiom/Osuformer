from typing import TypedDict


class PathsConfig(TypedDict):
    root: str
    data: str
    cache: str
    checkpoints: str
    logs: str
