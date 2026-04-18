import os
from pathlib import Path
from typing import Any

import requests

from .auth import TokenManager
from .beatmaps import BeatmapsEndpoint
from .tags import TagsEndpoint

_BASE_URL = "https://osu.ppy.sh/api/v2"
_DEFAULT_CACHE_PATH = Path(".cache/osu_token.json")


class OsuClient:
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        cache_path: Path = _DEFAULT_CACHE_PATH,
    ) -> None:
        cid = client_id or os.environ["OSU_CLIENT_ID"]
        csecret = client_secret or os.environ["OSU_CLIENT_SECRET"]
        self._token_manager = TokenManager(cid, csecret, cache_path)
        self._session = requests.Session()
        self.beatmaps = BeatmapsEndpoint(self)
        self.tags = TagsEndpoint(self)

    def get(self, path: str, **kwargs: Any) -> Any:
        token = self._token_manager.get_token()
        self._session.headers.update({"Authorization": f"Bearer {token}"})
        resp = self._session.get(f"{_BASE_URL}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()
