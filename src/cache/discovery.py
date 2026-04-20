from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


AUDIO_EXTENSIONS: tuple[str, ...] = (".mp3", ".ogg", ".wav", ".flac", ".aac", ".m4a")


@dataclass(frozen=True)
class BeatmapsetDir:
    set_id: int
    path: Path
    osu_files: list[Path]


def discover_beatmapsets(songs_root: Path) -> list[BeatmapsetDir]:
    out: list[BeatmapsetDir] = []
    if not songs_root.is_dir():
        return out
    for entry in sorted(songs_root.iterdir()):
        if not entry.is_dir():
            continue
        set_id = _parse_leading_int(entry.name)
        if set_id is None:
            continue
        osu_files = sorted(p for p in entry.glob("*.osu") if not p.name.startswith("._"))
        if not osu_files:
            continue
        out.append(BeatmapsetDir(set_id=set_id, path=entry, osu_files=osu_files))
    return out


def find_audio_file(beatmapset: BeatmapsetDir, audio_filename: str) -> Path | None:
    if audio_filename:
        candidate = beatmapset.path / audio_filename
        if candidate.exists() and not candidate.name.startswith("._"):
            return candidate
    for ext in AUDIO_EXTENSIONS:
        for candidate in beatmapset.path.glob(f"*{ext}"):
            if candidate.name.startswith("._"):
                continue
            return candidate
    return None


def _parse_leading_int(name: str) -> int | None:
    token = name.split(" ", 1)[0]
    return int(token) if token.isdigit() else None
