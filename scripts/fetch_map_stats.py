from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def load_env_file(path: Path) -> None:
    """Populate os.environ with keys from a simple KEY=VALUE .env file."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class OsuApiClient:
    def __init__(self, client_id: str, client_secret: str, base_url: str = "https://osu.ppy.sh", scope: str = "public"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.scope = scope
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    def _auth_headers(self) -> Dict[str, str]:
        token = self._get_token()
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def _get_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expiry:
            return self._token

        token_resp = requests.post(
            f"{self.base_url}/oauth/token",
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
                "scope": self.scope,
            },
            timeout=30,
        )
        token_resp.raise_for_status()
        payload = token_resp.json()
        self._token = payload["access_token"]
        expires_in = float(payload.get("expires_in", 0))
        self._token_expiry = now + max(0.0, expires_in - 30.0)
        return self._token

    def get_beatmap(self, beatmap_id: int) -> Dict:
        resp = requests.get(
            f"{self.base_url}/api/v2/beatmaps/{beatmap_id}",
            headers=self._auth_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def find_osu_files(dataset_path: Path) -> Iterable[Path]:
    return sorted(dataset_path.rglob("*.osu"))


def extract_ids(osu_file: Path) -> Tuple[Optional[int], Optional[int]]:
    beatmap_id = None
    beatmapset_id = None
    with osu_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("BeatmapID:"):
                try:
                    beatmap_id = int(line.split(":", 1)[1])
                except ValueError:
                    beatmap_id = None
            elif line.startswith("BeatmapSetID:"):
                try:
                    beatmapset_id = int(line.split(":", 1)[1])
                except ValueError:
                    beatmapset_id = None
            if beatmap_id is not None and beatmapset_id is not None:
                break
    return beatmap_id, beatmapset_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch osu! API stats for each difficulty in the dataset.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--dataset", default=None, help="Override dataset root (default: from config.paths.data).")
    parser.add_argument("--env-file", default=".env", help="Path to .env with OSU_CLIENT_ID/OSU_CLIENT_SECRET.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of .osu files processed.")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between API calls to respect rate limits.")
    parser.add_argument("--force", action="store_true", help="Re-fetch stats even if the .stats.json file exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))

    client_id = os.getenv("OSU_CLIENT_ID")
    client_secret = os.getenv("OSU_CLIENT_SECRET")
    api_base = os.getenv("OSU_API_BASE", "https://osu.ppy.sh")
    api_scope = os.getenv("OSU_API_SCOPE", "public")
    if not client_id or not client_secret:
        raise SystemExit("OSU_CLIENT_ID and OSU_CLIENT_SECRET must be set in the environment or .env file.")

    config = load_config(args.config)
    dataset_root = Path(args.dataset or config["paths"]["data"]).expanduser().resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset path '{dataset_root}' does not exist.")

    osu_files = list(find_osu_files(dataset_root))
    if not osu_files:
        print(f"[WARN] No .osu files found under {dataset_root}")
        return

    client = OsuApiClient(client_id, client_secret, api_base, api_scope)
    processed = 0
    skipped = 0
    for osu_path in osu_files:
        if args.limit is not None and processed >= args.limit:
            break

        beatmap_id, beatmapset_id = extract_ids(osu_path)
        if not beatmap_id:
            skipped += 1
            continue

        output_path = osu_path.with_suffix(".stats.json")
        if output_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            data = client.get_beatmap(beatmap_id)
        except requests.HTTPError as exc:
            print(f"[ERROR] Failed to fetch beatmap {beatmap_id} for '{osu_path}': {exc}")
            skipped += 1
            continue

        payload = {
            "beatmap_id": beatmap_id,
            "beatmapset_id": beatmapset_id,
            "source": f"{client.base_url}/api/v2/beatmaps/{beatmap_id}",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        processed += 1
        print(f"[INFO] Saved stats for beatmap {beatmap_id} → {output_path}")
        time.sleep(max(0.0, args.delay))

    print(f"[INFO] Done. saved={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
