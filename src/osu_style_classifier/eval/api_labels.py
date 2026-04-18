from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.osu_api import OsuClient


_MIN_REQUEST_INTERVAL_S = 1.1


class ApiLabelSource:
    def __init__(self, client: OsuClient, cache_dir: Path = Path(".cache/osu_beatmapsets")):
        self._client = client
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._tag_names = client.tags.by_id()
        self._last_request: float = 0.0

    def labels_for_set(self, set_id: int) -> dict[int, list[str]]:
        data = self._load_beatmapset(set_id)
        beatmaps: list[dict[str, Any]] = data.get("beatmaps", []) or []
        out: dict[int, list[str]] = {}
        for bm in beatmaps:
            if bm.get("mode_int") != 0:
                continue
            bm_id = int(bm["id"])
            tag_ids = bm.get("top_tag_ids") or []
            names: list[str] = []
            for entry in tag_ids:
                tid = entry.get("tag_id") if isinstance(entry, dict) else entry
                if tid is None:
                    continue
                tag = self._tag_names.get(int(tid))
                if tag is not None:
                    names.append(tag["name"])
            out[bm_id] = names
        return out

    def _load_beatmapset(self, set_id: int) -> dict[str, Any]:
        cache_path = self._cache_dir / f"{set_id}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        self._throttle()
        data = self._client.get(f"/beatmapsets/{set_id}")
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < _MIN_REQUEST_INTERVAL_S:
            time.sleep(_MIN_REQUEST_INTERVAL_S - elapsed)
        self._last_request = time.time()
