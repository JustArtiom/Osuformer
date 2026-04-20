from pathlib import Path

import numpy as np

from src.cache.audio import AudioFeature
from src.cache.maps import MapRecord
from src.cache.paths import CachePaths
from src.cache.reader import CacheReader
from src.cache.writer import AudioWriter, MapsWriter


def test_audio_roundtrip_single_file(tmp_path: Path) -> None:
    paths = CachePaths(tmp_path / "cache")
    paths.ensure()
    mel1 = (np.random.randn(100, 128) * 3.0).astype(np.float16)
    mel2 = (np.random.randn(50, 128) * 3.0).astype(np.float16)
    with AudioWriter(paths) as w:
        w.add(AudioFeature(key="aaa", mel=mel1, source_path=tmp_path / "a.mp3"))
        w.add(AudioFeature(key="bbb", mel=mel2, source_path=tmp_path / "b.mp3"))

    reader = CacheReader(tmp_path, "cache")
    recovered1 = reader.load_audio("aaa")
    recovered2 = reader.load_audio("bbb")
    assert np.array_equal(recovered1, mel1)
    assert np.array_equal(recovered2, mel2)


def test_audio_writer_is_idempotent(tmp_path: Path) -> None:
    paths = CachePaths(tmp_path / "cache")
    paths.ensure()
    mel = (np.random.randn(20, 128)).astype(np.float16)
    with AudioWriter(paths) as w:
        w.add(AudioFeature(key="same", mel=mel, source_path=tmp_path / "x.mp3"))
    with AudioWriter(paths) as w:
        assert w.has("same")
        w.add(AudioFeature(key="same", mel=mel, source_path=tmp_path / "x.mp3"))

    size_before = paths.audio_bin.stat().st_size
    with AudioWriter(paths) as w:
        w.add(AudioFeature(key="same", mel=mel, source_path=tmp_path / "x.mp3"))
    assert paths.audio_bin.stat().st_size == size_before


def test_maps_writer_binary_roundtrip(tmp_path: Path) -> None:
    paths = CachePaths(tmp_path / "cache")
    paths.ensure()
    mel = (np.random.randn(10, 128)).astype(np.float16)
    with AudioWriter(paths) as w:
        w.add(AudioFeature(key="a", mel=mel, source_path=tmp_path / "a.mp3"))

    record_a = MapRecord(
        beatmap_id=1,
        set_id=100,
        audio_key="a",
        osu_md5="hashA",
        version="Normal",
        creator="tester",
        mode=0,
        duration_ms=12345.0,
        circle_size=4.0,
        approach_rate=8.0,
        overall_difficulty=6.0,
        hp_drain_rate=5.0,
        slider_multiplier=1.4,
        primary_bpm=180.0,
        hitsounded=True,
        event_types=[1, 2, 3, 4, 5],
        event_values=[100, 200, 300, 400, 500],
        source_path=str(tmp_path / "a.osu"),
    )
    record_b = MapRecord(
        beatmap_id=2,
        set_id=100,
        audio_key="a",
        osu_md5="hashB",
        version="Hard",
        creator="tester",
        mode=0,
        duration_ms=6789.0,
        circle_size=4.2,
        approach_rate=9.0,
        overall_difficulty=7.0,
        hp_drain_rate=6.0,
        slider_multiplier=1.8,
        primary_bpm=200.0,
        hitsounded=False,
        event_types=[10, 20],
        event_values=[1000, 2000],
        source_path=str(tmp_path / "b.osu"),
    )
    with MapsWriter(paths) as w:
        w.add(record_a)
        w.add(record_b)

    reader = CacheReader(tmp_path, "cache")
    got_a = reader.load_map(1)
    got_b = reader.load_map(2)

    assert list(got_a["event_types"]) == record_a.event_types
    assert list(got_a["event_values"]) == record_a.event_values
    assert got_a["version"] == "Normal"
    assert got_a["hitsounded"] is True
    assert list(got_b["event_types"]) == record_b.event_types
    assert list(got_b["event_values"]) == record_b.event_values
    assert got_b["version"] == "Hard"
