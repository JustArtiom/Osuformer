from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .types import Beatmap, BeatmapSearchResult, Beatmapset

if TYPE_CHECKING:
    from .client import OsuClient

RankedStatus = Literal["ranked", "approved", "qualified", "loved", "pending", "graveyard", "any"]
GameMode = Literal["osu", "taiko", "fruits", "mania"]


class BeatmapsEndpoint:
    def __init__(self, client: OsuClient) -> None:
        self._client = client

    def get(self, beatmap_id: int) -> Beatmap:
        return self._client.get(f"/beatmaps/{beatmap_id}")

    def get_beatmapset(self, beatmapset_id: int) -> Beatmapset:
        return self._client.get(f"/beatmapsets/{beatmapset_id}")

    def search(
        self,
        query: str = "",
        mode: GameMode | None = None,
        status: RankedStatus | None = None,
        cursor_string: str | None = None,
    ) -> BeatmapSearchResult:
        params: dict[str, str] = {}
        if query:
            params["q"] = query
        if mode is not None:
            params["m"] = mode
        if status is not None:
            params["s"] = status
        if cursor_string is not None:
            params["cursor_string"] = cursor_string
        return self._client.get("/beatmapsets/search", params=params)
