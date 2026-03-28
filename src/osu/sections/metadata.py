from typing import List
import re

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
    tags: List[str] | None = None,
    beatmap_id: int = 0,
    beatmap_set_id: int = 0,
  ):
    self.title = title
    self.title_unicode = title_unicode
    self.artist = artist
    self.artist_unicode = artist_unicode
    self.creator = creator
    self.version = version
    self.source = source
    self.tags = list(tags) if tags is not None else []
    self.beatmap_id = beatmap_id
    self.beatmap_set_id = beatmap_set_id

    if raw:
      kv = self.key_value(raw)
      for k, v in kv.items():
        attr = self._normalize_key(k)
        if hasattr(self, attr):
          current_value = getattr(self, attr)
          try:
            if attr == "tags":
              setattr(self, attr, [tag.strip() for tag in v.split(" ") if tag.strip()])
            else:
              setattr(self, attr, type(current_value)(v))
          except (TypeError, ValueError):
            setattr(self, attr, v)
  
  def key_value(self, raw: str) -> dict[str, str]:
    rows = raw.splitlines()
    key_value_pairs = []
    for row in rows:
      if ":" not in row:
        continue
      key, value = (segment.strip() for segment in row.split(":", 1))
      key_value_pairs.append((key, value))
    return dict(key_value_pairs)
  
  def _normalize_key(self, key: str) -> str:
    key = key.strip().replace(" ", "_")
    key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return key.lower()
  
  def __str__(self) -> str:
    return (
      f"Title:{self.title}\n"
      f"TitleUnicode:{self.title_unicode}\n"
      f"Artist:{self.artist}\n"
      f"ArtistUnicode:{self.artist_unicode}\n"
      f"Creator:{self.creator}\n"
      f"Version:{self.version}\n"
      f"Source:{self.source}\n"
      f"Tags:{' '.join(self.tags)}\n"
      f"BeatmapID:{self.beatmap_id}\n"
      f"BeatmapSetID:{self.beatmap_set_id}"
    )
