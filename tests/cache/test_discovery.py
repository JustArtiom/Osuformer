from pathlib import Path

from src.cache.discovery import discover_beatmapsets, find_audio_file


def test_discover_skips_non_numeric_and_empty(tmp_path: Path) -> None:
    (tmp_path / "1000 Song A").mkdir()
    (tmp_path / "notaset").mkdir()
    (tmp_path / "2000 Song B").mkdir()
    (tmp_path / "3000 Empty Song").mkdir()
    (tmp_path / "1000 Song A" / "song.osu").write_text("dummy")
    (tmp_path / "2000 Song B" / "song.osu").write_text("dummy")

    sets = discover_beatmapsets(tmp_path)
    ids = [s.set_id for s in sets]
    assert ids == [1000, 2000]


def test_discover_skips_applesupport_shadow_files(tmp_path: Path) -> None:
    d = tmp_path / "500 Test"
    d.mkdir()
    (d / "real.osu").write_text("dummy")
    (d / "._real.osu").write_text("apple shadow")

    sets = discover_beatmapsets(tmp_path)
    assert len(sets) == 1
    assert all(not p.name.startswith("._") for p in sets[0].osu_files)


def test_find_audio_file_falls_back_to_glob(tmp_path: Path) -> None:
    d = tmp_path / "1 Test"
    d.mkdir()
    (d / "song.osu").write_text("")
    (d / "audio.mp3").write_text("")
    sets = discover_beatmapsets(tmp_path)
    assert find_audio_file(sets[0], "") == d / "audio.mp3"
