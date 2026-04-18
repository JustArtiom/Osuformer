from src.osu_style_classifier import classify

from ..fixtures import build_beatmap, make_circle, make_timing_point


def test_simple_rhythm_triggers_expression_simple() -> None:
    tp = make_timing_point(bpm=120.0)
    interval = 60000.0 / 120.0 / 2
    circles = [make_circle(1000.0 + i * interval, x=100.0 + i * 2, y=192) for i in range(60)]
    beatmap = build_beatmap(circles, timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "expression/simple" in tags


def test_time_signatures_triggers_when_non_4_4() -> None:
    tps = [make_timing_point(bpm=180.0, meter=7)]
    circles = [make_circle(1000.0 + i * 500.0) for i in range(20)]
    beatmap = build_beatmap(circles, timing_points=tps)

    tags = set(classify(beatmap).tags)
    assert "meta/time signatures" in tags
