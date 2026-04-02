import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.osu_api.auth import TokenManager
from src.osu_api.types import TokenCache


def _manager(tmp_path: Path) -> TokenManager:
    return TokenManager("client_id", "client_secret", tmp_path / "token.json")


def _mock_post_response(token: str = "test_token", expires_in: int = 86400) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"access_token": token, "expires_in": expires_in}
    return resp


def test_get_token_fetches_when_no_cache(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with patch("requests.post", return_value=_mock_post_response()) as mock_post:
        token = manager.get_token()
    assert token == "test_token"
    mock_post.assert_called_once()


def test_get_token_uses_valid_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "token.json"
    cache: TokenCache = {"access_token": "cached_token", "expires_at": time.time() + 3600}
    cache_path.write_text(json.dumps(cache))
    manager = TokenManager("client_id", "client_secret", cache_path)
    with patch("requests.post") as mock_post:
        token = manager.get_token()
    assert token == "cached_token"
    mock_post.assert_not_called()


def test_get_token_refetches_when_expired(tmp_path: Path) -> None:
    cache_path = tmp_path / "token.json"
    cache: TokenCache = {"access_token": "old_token", "expires_at": time.time() - 10}
    cache_path.write_text(json.dumps(cache))
    manager = TokenManager("client_id", "client_secret", cache_path)
    with patch("requests.post", return_value=_mock_post_response("new_token")):
        token = manager.get_token()
    assert token == "new_token"


def test_fetch_writes_cache_file(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with patch("requests.post", return_value=_mock_post_response("abc123", expires_in=3600)):
        manager.get_token()
    data = json.loads((tmp_path / "token.json").read_text())
    assert data["access_token"] == "abc123"
    assert data["expires_at"] > time.time()
    assert data["expires_at"] < time.time() + 3601


def test_fetch_sends_correct_credentials(tmp_path: Path) -> None:
    manager = TokenManager("my_id", "my_secret", tmp_path / "token.json")
    with patch("requests.post", return_value=_mock_post_response()) as mock_post:
        manager.get_token()
    sent_data = mock_post.call_args.kwargs["data"]
    assert sent_data["client_id"] == "my_id"
    assert sent_data["client_secret"] == "my_secret"
    assert sent_data["grant_type"] == "client_credentials"
    assert sent_data["scope"] == "public"


def test_cache_directory_created_if_missing(tmp_path: Path) -> None:
    cache_path = tmp_path / "nested" / "dir" / "token.json"
    manager = TokenManager("id", "secret", cache_path)
    with patch("requests.post", return_value=_mock_post_response()):
        manager.get_token()
    assert cache_path.exists()
