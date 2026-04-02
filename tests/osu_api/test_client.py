import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.osu_api.client import OsuClient


def _client_with_cached_token(tmp_path: Path, token: str = "test_token") -> OsuClient:
    cache_path = tmp_path / "token.json"
    cache_path.write_text(json.dumps({"access_token": token, "expires_at": time.time() + 3600}))
    return OsuClient("id", "secret", cache_path=cache_path)


def test_client_reads_client_id_from_env(tmp_path: Path) -> None:
    env = {"OSU_CLIENT_ID": "env_id", "OSU_CLIENT_SECRET": "env_secret"}
    with patch.dict(os.environ, env):
        client = OsuClient(cache_path=tmp_path / "token.json")
    assert client._token_manager._client_id == "env_id"
    assert client._token_manager._client_secret == "env_secret"


def test_client_raises_when_env_missing(tmp_path: Path) -> None:
    env = {k: "" for k in ("OSU_CLIENT_ID", "OSU_CLIENT_SECRET")}
    stripped = {k: v for k, v in os.environ.items() if k not in env}
    with patch.dict(os.environ, stripped, clear=True):
        with pytest.raises(KeyError):
            OsuClient(cache_path=tmp_path / "token.json")


def test_get_sets_bearer_header(tmp_path: Path) -> None:
    client = _client_with_cached_token(tmp_path, token="my_token")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"id": 1}
    with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
        client.get("/beatmaps/1")
    headers = client._session.headers
    assert headers["Authorization"] == "Bearer my_token"


def test_get_calls_correct_url(tmp_path: Path) -> None:
    client = _client_with_cached_token(tmp_path)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {}
    with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
        client.get("/beatmaps/42")
    mock_get.assert_called_once_with("https://osu.ppy.sh/api/v2/beatmaps/42")


def test_get_raises_on_http_error(tmp_path: Path) -> None:
    client = _client_with_cached_token(tmp_path)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")
    with patch.object(client._session, "get", return_value=mock_resp):
        with pytest.raises(Exception, match="401"):
            client.get("/beatmaps/1")


def test_client_has_beatmaps_endpoint(tmp_path: Path) -> None:
    client = _client_with_cached_token(tmp_path)
    from src.osu_api.beatmaps import BeatmapsEndpoint
    assert isinstance(client.beatmaps, BeatmapsEndpoint)
