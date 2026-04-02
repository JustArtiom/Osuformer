import os
from unittest.mock import MagicMock

import pytest

from src.osu_api.beatmaps import BeatmapsEndpoint
from src.osu_api.client import OsuClient

_requires_token = pytest.mark.skipif(
    not os.environ.get("OSU_CLIENT_ID"),
    reason="OSU_CLIENT_ID not set",
)


def _endpoint() -> tuple[BeatmapsEndpoint, MagicMock]:
    client = MagicMock(spec=OsuClient)
    return BeatmapsEndpoint(client), client


def test_get_calls_correct_path() -> None:
    endpoint, client = _endpoint()
    endpoint.get(123)
    client.get.assert_called_once_with("/beatmaps/123")


def test_get_beatmapset_calls_correct_path() -> None:
    endpoint, client = _endpoint()
    endpoint.get_beatmapset(456)
    client.get.assert_called_once_with("/beatmapsets/456")


def test_search_no_params_sends_empty_dict() -> None:
    endpoint, client = _endpoint()
    endpoint.search()
    client.get.assert_called_once_with("/beatmapsets/search", params={})


def test_search_passes_all_params() -> None:
    endpoint, client = _endpoint()
    endpoint.search(query="songs", mode="osu", status="ranked", cursor_string="abc")
    client.get.assert_called_once_with(
        "/beatmapsets/search",
        params={"q": "songs", "m": "osu", "s": "ranked", "cursor_string": "abc"},
    )


def test_search_omits_none_params() -> None:
    endpoint, client = _endpoint()
    endpoint.search(query="test")
    params = client.get.call_args.kwargs["params"]
    assert "m" not in params
    assert "s" not in params
    assert "cursor_string" not in params


def test_search_omits_empty_query() -> None:
    endpoint, client = _endpoint()
    endpoint.search(mode="taiko")
    params = client.get.call_args.kwargs["params"]
    assert "q" not in params
    assert params["m"] == "taiko"


@_requires_token
def test_integration_search_ranked() -> None:
    from src.osu_api import OsuClient
    client = OsuClient()
    result = client.beatmaps.search(status="ranked", mode="osu")
    assert "beatmapsets" in result
    assert len(result["beatmapsets"]) > 0


@_requires_token
def test_integration_get_beatmapset() -> None:
    from src.osu_api import OsuClient
    client = OsuClient()
    beatmapset = client.beatmaps.get_beatmapset(241526)
    assert beatmapset["id"] == 241526
    assert "beatmaps" in beatmapset
