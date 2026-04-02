from typing import NotRequired, TypedDict


class TokenCache(TypedDict):
    access_token: str
    expires_at: float


class BeatmapCovers(TypedDict):
    cover: str
    card: str
    list: str
    slimcover: str


class Beatmap(TypedDict):
    id: int
    beatmapset_id: int
    mode: str
    mode_int: int
    status: str
    version: str
    difficulty_rating: float
    bpm: float
    cs: float
    ar: float
    accuracy: float
    drain: float
    total_length: int
    hit_length: int
    url: str
    checksum: NotRequired[str]
    max_combo: NotRequired[int]


class UserTag(TypedDict):
    id: int
    name: str
    description: str
    ruleset_id: NotRequired[int | None]


class Beatmapset(TypedDict):
    id: int
    artist: str
    artist_unicode: str
    title: str
    title_unicode: str
    creator: str
    user_id: int
    tags: str
    status: str
    bpm: float
    submitted_date: NotRequired[str]
    ranked_date: NotRequired[str]
    covers: BeatmapCovers
    beatmaps: NotRequired[list[Beatmap]]
    related_tags: NotRequired[list[UserTag]]


class BeatmapSearchResult(TypedDict):
    beatmapsets: list[Beatmapset]
    cursor_string: NotRequired[str]
