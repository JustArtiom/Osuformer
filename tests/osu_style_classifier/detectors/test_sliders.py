from src.osu_style_classifier import classify

from ..fixtures import build_beatmap, make_inherited_tp, make_slider, make_timing_point


def test_slider_only_when_mostly_sliders() -> None:
    tp = make_timing_point(bpm=180.0)
    sliders = [make_slider(time=1000.0 + 250.0 * i, x=100.0 + 10.0 * i, y=200.0) for i in range(40)]
    beatmap = build_beatmap(sliders, timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "gimmick/slider only" in tags


def test_high_sv_triggers_high_sv_tag() -> None:
    tps = [make_timing_point(bpm=180.0), make_inherited_tp(1000.0, sv_multiplier=2.0)]
    sliders = [make_slider(time=1000.0 + 250.0 * i, x=100.0 + 10.0 * i, y=200.0) for i in range(10)]
    beatmap = build_beatmap(sliders, timing_points=tps)

    tags = set(classify(beatmap).tags)
    assert "sliders/high sv" in tags
