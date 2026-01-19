from typing import List

class Metadata:
  def __init__(
    self, *,
    raw: str = "",
    title: str = "Unknown",
    title_unicode: str = "Unknown",
    artist: str = "Unknown",
    artist_unicode: str = "Unknown",
    creator: str = "Unknown",
    version: str = "Unknown",
    source: str = "",
    tags: List[str] = [],
    beatmap_id: int = 0,
    beatmap_set_id: int = 0,
  ):
    pass