import re

from ..utils import fmt


class Editor:
    def __init__(
        self,
        *,
        raw: str = "",
        bookmarks: list[int] | None = None,
        distance_spacing: float = 1.0,
        beat_divisor: int = 4,
        grid_size: int = 32,
        timeline_zoom: float = 1.0,
    ):
        self.bookmarks = list(bookmarks) if bookmarks is not None else []
        self.distance_spacing = distance_spacing
        self.beat_divisor = beat_divisor
        self.grid_size = grid_size
        self.timeline_zoom = timeline_zoom

        if raw:
            kv = self._key_value(raw)
            for k, v in kv.items():
                attr = self._normalize_key(k)
                if hasattr(self, attr):
                    current_value = getattr(self, attr)
                    try:
                        if attr == "bookmarks":
                            setattr(self, attr, [int(b) for b in v.split(",") if b.strip().isdigit()])
                        else:
                            setattr(self, attr, type(current_value)(v))
                    except (TypeError, ValueError):
                        setattr(self, attr, v)

    def _key_value(self, raw: str) -> dict[str, str]:
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
            f"Bookmarks: {','.join(str(b) for b in self.bookmarks)}\n"
            f"DistanceSpacing: {fmt(self.distance_spacing)}\n"
            f"BeatDivisor: {self.beat_divisor}\n"
            f"GridSize: {self.grid_size}\n"
            f"TimelineZoom: {fmt(self.timeline_zoom)}"
        )
