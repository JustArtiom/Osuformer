from src.osu_style_classifier import classify

from ..fixtures import build_beatmap, make_circle, make_spinner, make_timing_point


def test_ninja_spinners_detected_on_short_spinners() -> None:
    tp = make_timing_point(bpm=180.0)
    objs: list = []
    for i in range(4):
        start = 1000.0 + i * 2000.0
        objs.append(make_spinner(time=start, end_time=start + 300.0))
    objs.extend(make_circle(500.0 + i * 250.0) for i in range(5))
    beatmap = build_beatmap(objs, timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "gimmick/ninja spinners" in tags
