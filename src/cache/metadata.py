from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.osu_api import OsuClient
from src.osu_tokenizer import DESCRIPTOR_TO_INDEX

from .paths import CachePaths


_MIN_REQUEST_INTERVAL_S = 1.1
_MAX_RETRIES = 5
_BACKOFF_BASE_S = 2.0
_BACKOFF_CAP_S = 60.0


@dataclass(frozen=True)
class MetadataRecord:
    beatmap_id: int
    star_rating: float
    ranked_year: int
    descriptor_indices: list[int]


class MetadataFetcher:
    def __init__(self, client: OsuClient, cache_dir: Path):
        self._client = client
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request: float = 0.0
        self._tag_index: dict[int, str] = {}
        self._tag_index_loaded = False

    def load_tags(self) -> dict[int, str]:
        if self._tag_index_loaded:
            return self._tag_index
        cache_path = self._cache_dir / "tags.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
        else:
            data = self._get_with_retry("/tags")
            if data is None:
                raise RuntimeError("failed to fetch /tags after retries")
            cache_path.write_text(json.dumps(data))
        self._tag_index = {int(t["id"]): str(t["name"]) for t in data.get("tags", [])}
        self._tag_index_loaded = True
        return self._tag_index

    def fetch_set(self, set_id: int) -> dict[str, Any] | None:
        cache_path = self._cache_dir / f"{set_id}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                cache_path.unlink(missing_ok=True)
        data = self._get_with_retry(f"/beatmapsets/{set_id}")
        if data is None:
            return None
        cache_path.write_text(json.dumps(data))
        return data

    def _get_with_retry(self, path: str) -> dict[str, Any] | None:
        for attempt in range(_MAX_RETRIES):
            self._throttle()
            try:
                return self._client.get(path)
            except requests.HTTPError as e:
                resp = e.response
                status = resp.status_code if resp is not None else None
                if status == 404:
                    return None
                if status == 429:
                    retry_after = _parse_retry_after(resp)
                    wait = retry_after if retry_after is not None else _backoff_seconds(attempt)
                    print(f"  rate limited on {path}, sleeping {wait:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})", flush=True)
                    time.sleep(wait)
                    continue
                if status is not None and 500 <= status < 600:
                    wait = _backoff_seconds(attempt)
                    print(f"  {status} on {path}, sleeping {wait:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})", flush=True)
                    time.sleep(wait)
                    continue
                return None
            except (requests.ConnectionError, requests.Timeout):
                wait = _backoff_seconds(attempt)
                print(f"  connection error on {path}, sleeping {wait:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})", flush=True)
                time.sleep(wait)
                continue
            except Exception:
                return None
        return None

    def extract_records(self, set_id: int, wanted_beatmap_ids: set[int]) -> list[MetadataRecord]:
        data = self.fetch_set(set_id)
        if data is None:
            return []
        tags = self.load_tags()
        ranked_date = data.get("ranked_date") or ""
        year = _parse_year(ranked_date)
        out: list[MetadataRecord] = []
        for bm in data.get("beatmaps") or []:
            bm_id = int(bm.get("id", 0))
            if bm_id not in wanted_beatmap_ids:
                continue
            if bm.get("mode_int") != 0:
                continue
            star_rating = float(bm.get("difficulty_rating", 0.0))
            top_tags = bm.get("top_tag_ids") or []
            descriptor_indices: list[int] = []
            for entry in top_tags:
                tag_id = entry.get("tag_id") if isinstance(entry, dict) else entry
                if tag_id is None:
                    continue
                name = tags.get(int(tag_id))
                if name is None:
                    continue
                idx = DESCRIPTOR_TO_INDEX.get(name)
                if idx is not None:
                    descriptor_indices.append(idx)
            out.append(
                MetadataRecord(
                    beatmap_id=bm_id,
                    star_rating=star_rating,
                    ranked_year=year,
                    descriptor_indices=descriptor_indices,
                )
            )
        return out

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < _MIN_REQUEST_INTERVAL_S:
            time.sleep(_MIN_REQUEST_INTERVAL_S - elapsed)
        self._last_request = time.time()


def _parse_year(ranked_date: str) -> int:
    if not ranked_date:
        return 0
    try:
        return int(ranked_date[:4])
    except Exception:
        return 0


def _parse_retry_after(resp: requests.Response | None) -> float | None:
    if resp is None:
        return None
    header = resp.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _backoff_seconds(attempt: int) -> float:
    return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2**attempt))


def write_metadata(paths: CachePaths, records: list[MetadataRecord]) -> None:
    if not records:
        return
    existing: list[dict] = []
    if paths.metadata.exists():
        existing = pq.read_table(paths.metadata).to_pylist()
    seen = {int(r["beatmap_id"]) for r in existing}
    for r in records:
        if r.beatmap_id in seen:
            continue
        existing.append(
            {
                "beatmap_id": r.beatmap_id,
                "star_rating": r.star_rating,
                "ranked_year": r.ranked_year,
                "descriptor_indices": r.descriptor_indices,
            }
        )
        seen.add(r.beatmap_id)
    pq.write_table(pa.Table.from_pylist(existing), paths.metadata)


def read_metadata(paths: CachePaths) -> dict[int, MetadataRecord]:
    out: dict[int, MetadataRecord] = {}
    if not paths.metadata.exists():
        return out
    for row in pq.read_table(paths.metadata).to_pylist():
        out[int(row["beatmap_id"])] = MetadataRecord(
            beatmap_id=int(row["beatmap_id"]),
            star_rating=float(row["star_rating"]),
            ranked_year=int(row["ranked_year"]),
            descriptor_indices=list(row["descriptor_indices"]),
        )
    return out
