import json
import time
from pathlib import Path

import requests

from .types import TokenCache

_TOKEN_URL = "https://osu.ppy.sh/oauth/token"
_TOKEN_EXPIRY_BUFFER = 60


class TokenManager:
    def __init__(self, client_id: str, client_secret: str, cache_path: Path) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._cache_path = cache_path

    def get_token(self) -> str:
        cached = self._load_cache()
        if cached is not None and cached["expires_at"] > time.time() + _TOKEN_EXPIRY_BUFFER:
            return cached["access_token"]
        return self._fetch_and_cache()

    def _load_cache(self) -> TokenCache | None:
        if not self._cache_path.exists():
            return None
        with open(self._cache_path) as f:
            return json.load(f)

    def _fetch_and_cache(self) -> str:
        resp = requests.post(
            _TOKEN_URL,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "client_credentials",
                "scope": "public",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        cache: TokenCache = {
            "access_token": data["access_token"],
            "expires_at": time.time() + data["expires_in"],
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(cache, f)
        return cache["access_token"]
