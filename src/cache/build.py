from __future__ import annotations

import signal
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.config.schemas.audio import AudioConfig
from src.config.schemas.tokenizer import TokenizerConfig
from src.osu.beatmap import Beatmap
from src.osu_tokenizer import Vocab

from .audio import compute_audio_feature, hash_audio_file
from .discovery import BeatmapsetDir, discover_beatmapsets, find_audio_file
from .maps import parse_and_tokenize
from .paths import CachePaths
from .writer import AudioWriter, MapsWriter


class _PerSetTimeout(Exception):
    pass


def _install_alarm(seconds: int) -> bool:
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return False

    def _handler(signum, frame):
        raise _PerSetTimeout(f"set processing exceeded {seconds}s")

    signal.signal(signal.SIGALRM, _handler)
    return True


@dataclass
class BuildStats:
    sets_seen: int = 0
    sets_skipped: int = 0
    maps_written: int = 0
    maps_skipped: int = 0
    audios_written: int = 0
    audios_reused: int = 0
    errors: int = 0


def build_cache(
    songs_root: Path,
    cache_root: Path,
    name: str,
    audio_cfg: AudioConfig,
    tokenizer_cfg: TokenizerConfig,
    limit: int | None = None,
    set_timeout_s: int = 120,
) -> BuildStats:
    paths = CachePaths(root=cache_root / name)
    paths.ensure()
    vocab = Vocab(tokenizer_cfg)
    stats = BuildStats()

    sets = discover_beatmapsets(songs_root)
    if limit is not None:
        sets = sets[:limit]

    alarm_enabled = _install_alarm(set_timeout_s)

    with AudioWriter(paths) as audio_writer, MapsWriter(paths) as maps_writer:
        for beatmapset in tqdm(sets, desc="sets"):
            stats.sets_seen += 1
            if alarm_enabled:
                signal.alarm(set_timeout_s)
            try:
                _process_set(beatmapset, audio_writer, maps_writer, audio_cfg, tokenizer_cfg, vocab, stats)
            except _PerSetTimeout:
                stats.errors += 1
                tqdm.write(f"  timeout on set {beatmapset.set_id} at {beatmapset.path}")
                continue
            except Exception:
                stats.errors += 1
                continue
            finally:
                if alarm_enabled:
                    signal.alarm(0)
        audio_writer.flush_index()
        maps_writer.flush()

    return stats


def _process_set(
    beatmapset: BeatmapsetDir,
    audio_writer: AudioWriter,
    maps_writer: MapsWriter,
    audio_cfg: AudioConfig,
    tokenizer_cfg: TokenizerConfig,
    vocab: Vocab,
    stats: BuildStats,
) -> None:
    first_audio_filename = _detect_audio_filename(beatmapset.osu_files[0])
    audio_path = find_audio_file(beatmapset, first_audio_filename)
    if audio_path is None or not audio_path.exists():
        stats.sets_skipped += 1
        return

    audio_key = hash_audio_file(audio_path)
    if not audio_writer.has(audio_key):
        try:
            feature = compute_audio_feature(audio_path, audio_cfg)
            audio_writer.add(feature)
            stats.audios_written += 1
        except Exception:
            stats.sets_skipped += 1
            return
    else:
        stats.audios_reused += 1

    for osu_path in beatmapset.osu_files:
        try:
            beatmap = Beatmap(file_path=str(osu_path))
        except Exception:
            stats.errors += 1
            continue
        if beatmap.general.mode.value != 0:
            continue
        if maps_writer.has(beatmap.metadata.beatmap_id):
            stats.maps_skipped += 1
            continue
        record = parse_and_tokenize(
            osu_path=osu_path,
            set_id=beatmapset.set_id,
            audio_key=audio_key,
            tokenizer_cfg=tokenizer_cfg,
            vocab=vocab,
        )
        if record is None:
            stats.errors += 1
            continue
        maps_writer.add(record)
        stats.maps_written += 1


def _detect_audio_filename(osu_path: Path) -> str:
    try:
        beatmap = Beatmap(file_path=str(osu_path))
    except Exception:
        return ""
    return beatmap.general.audio_filename
