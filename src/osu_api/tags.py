from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .client import OsuClient


class Tag(TypedDict):
    id: int
    name: str
    ruleset_id: int | None
    description: str


class TagsResponse(TypedDict):
    tags: list[Tag]


class TagsEndpoint:
    def __init__(self, client: "OsuClient") -> None:
        self._client = client
        self._cache: dict[int, Tag] | None = None

    def list(self) -> list[Tag]:
        data: TagsResponse = self._client.get("/tags")
        return data["tags"]

    def by_id(self) -> dict[int, Tag]:
        if self._cache is None:
            self._cache = {t["id"]: t for t in self.list()}
        return self._cache

    def name_for(self, tag_id: int) -> str | None:
        tag = self.by_id().get(tag_id)
        return tag["name"] if tag is not None else None
