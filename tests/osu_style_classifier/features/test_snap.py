from src.osu_style_classifier.features.snap import classify_snap


def test_classify_snap_quarter() -> None:
    bpm = 180.0
    beat_length = 60000.0 / bpm
    assert classify_snap(beat_length / 4, beat_length) == 4


def test_classify_snap_triplet() -> None:
    bpm = 180.0
    beat_length = 60000.0 / bpm
    assert classify_snap(beat_length / 3, beat_length) == 3


def test_classify_snap_returns_none_on_unsnapped() -> None:
    bpm = 180.0
    beat_length = 60000.0 / bpm
    assert classify_snap(beat_length * 1.37, beat_length, tolerance_ms=2.0) is None
