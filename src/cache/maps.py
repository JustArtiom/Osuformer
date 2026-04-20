from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.config.schemas.tokenizer import TokenizerConfig
from src.osu.beatmap import Beatmap
from src.osu.enums import HitSound
from src.osu_tokenizer import EventType, Vocab, beatmap_to_events, collect_timing_events, merge_by_time


@dataclass(frozen=True)
class MapRecord:
    beatmap_id: int
    set_id: int
    audio_key: str
    osu_md5: str
    version: str
    creator: str
    mode: int
    duration_ms: float
    circle_size: float
    approach_rate: float
    overall_difficulty: float
    hp_drain_rate: float
    slider_multiplier: float
    primary_bpm: float
    hitsounded: bool
    event_types: list[int]
    event_values: list[int]
    source_path: str


def parse_and_tokenize(
    osu_path: Path,
    set_id: int,
    audio_key: str,
    tokenizer_cfg: TokenizerConfig,
    vocab: Vocab,
) -> MapRecord | None:
    try:
        beatmap = Beatmap(file_path=str(osu_path))
    except Exception:
        return None
    if beatmap.general.mode.value != 0:
        return None
    if not beatmap.hit_objects:
        return None

    start_ms = 0.0
    stream = beatmap_to_events(beatmap, window_start_ms=start_ms, vocab=vocab, config=tokenizer_cfg, clamp_abs_time=False)
    timing = collect_timing_events(beatmap, start_ms, vocab, tokenizer_cfg, clamp_abs_time=False)
    merged = merge_by_time(stream.events, timing)

    type_order = _type_list(vocab)
    type_to_idx = {t: i for i, t in enumerate(type_order)}
    event_types = [type_to_idx[e.type] for e in merged]
    event_values = [e.value for e in merged]
    if any(v > 2_147_483_647 or v < -2_147_483_648 for v in event_values):
        return None

    duration_ms = float(beatmap.hit_objects[-1].time) - float(beatmap.hit_objects[0].time)
    primary_bpm = _compute_primary_bpm(beatmap)
    hitsounded = _is_hitsounded(beatmap)
    osu_md5 = _hash_file(osu_path)

    return MapRecord(
        beatmap_id=beatmap.metadata.beatmap_id,
        set_id=set_id,
        audio_key=audio_key,
        osu_md5=osu_md5,
        version=beatmap.metadata.version,
        creator=beatmap.metadata.creator,
        mode=int(beatmap.general.mode.value),
        duration_ms=duration_ms,
        circle_size=float(beatmap.difficulty.circle_size),
        approach_rate=float(beatmap.difficulty.approach_rate),
        overall_difficulty=float(beatmap.difficulty.overall_difficulty),
        hp_drain_rate=float(beatmap.difficulty.hp_drain_rate),
        slider_multiplier=float(beatmap.difficulty.slider_multiplier),
        primary_bpm=primary_bpm,
        hitsounded=hitsounded,
        event_types=event_types,
        event_values=event_values,
        source_path=str(osu_path),
    )


def _type_list(vocab: Vocab) -> list[EventType]:
    return [er.type for er in vocab.output_ranges] + [er.type for er in vocab.input_ranges]


def _compute_primary_bpm(beatmap: Beatmap) -> float:
    uninherited = [tp for tp in beatmap.timing_points if tp.uninherited == 1 and tp.beat_length > 0]
    if not uninherited:
        return 0.0
    return 60000.0 / uninherited[0].beat_length


def _is_hitsounded(beatmap: Beatmap) -> bool:
    for obj in beatmap.hit_objects:
        if obj.hit_sound != HitSound.NONE and obj.hit_sound != HitSound.NORMAL:
            return True
        sample = getattr(obj, "hit_sample", None)
        if sample is not None and getattr(sample, "filename", ""):
            return True
    return False


def _hash_file(path: Path, chunk_size: int = 1 << 16) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
