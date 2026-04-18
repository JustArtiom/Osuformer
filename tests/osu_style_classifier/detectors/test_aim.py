from src.osu_style_classifier import classify

from ..fixtures import build_beatmap, jump_circles, make_timing_point


def test_detects_jumps_at_large_spacing() -> None:
    tp = make_timing_point(bpm=180.0)
    beatmap = build_beatmap(jump_circles(count=80, start_ms=1000.0, interval_ms=330.0), timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "skillset/jumps" in tags


def test_detects_sharp_aim_on_alternating_corners() -> None:
    tp = make_timing_point(bpm=180.0)
    beatmap = build_beatmap(jump_circles(count=80, start_ms=1000.0, interval_ms=330.0), timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "jumps/sharp" in tags or "jumps/wide" in tags
