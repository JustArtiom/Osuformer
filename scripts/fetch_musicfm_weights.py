from __future__ import annotations

import hashlib
from pathlib import Path

import click
import requests

_HF_BASE = "https://huggingface.co/minzwon/MusicFM/resolve/main"

_VARIANTS: dict[str, dict[str, str]] = {
    "msd": {
        "stats": "msd_stats.json",
        "weights": "pretrained_msd.pt",
    },
    "fma": {
        "stats": "fma_stats.json",
        "weights": "pretrained_fma.pt",
    },
}


@click.command()
@click.option(
    "--variant",
    type=click.Choice(sorted(_VARIANTS.keys())),
    default="msd",
    help="MusicFM checkpoint variant. msd is recommended (better than fma).",
)
@click.option(
    "--dest",
    type=click.Path(path_type=Path),
    required=True,
    help="Destination directory (created if missing).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-download even if files already exist.",
)
def main(variant: str, dest: Path, force: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    files = _VARIANTS[variant]
    for label, filename in files.items():
        url = f"{_HF_BASE}/{filename}"
        target = dest / filename
        if target.exists() and not force:
            print(f"[skip] {label}: {target} already exists ({_human_bytes(target.stat().st_size)})")
            continue
        print(f"[fetch] {label}: {url}")
        _download(url, target)
        size = target.stat().st_size
        sha = _sha256(target)
        print(f"        wrote {target} ({_human_bytes(size)}, sha256={sha[:16]}...)")
    print("done")


def _download(url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", "0"))
        seen = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                seen += len(chunk)
                if total > 0:
                    pct = 100.0 * seen / total
                    print(f"  {pct:5.1f}%  {_human_bytes(seen)}/{_human_bytes(total)}", end="\r", flush=True)
        if total > 0:
            print()
    tmp.rename(target)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n //= 1024
    return f"{n:.1f}TB"


if __name__ == "__main__":
    main()
