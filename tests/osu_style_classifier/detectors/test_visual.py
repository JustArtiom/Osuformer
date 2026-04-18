from src.osu_style_classifier import classify

from ..fixtures import build_beatmap, make_circle, make_timing_point


def test_grid_snap_when_positions_on_grid() -> None:
    tp = make_timing_point(bpm=180.0)
    positions = [(16 * (i % 16), 16 * ((i // 16) % 12)) for i in range(60)]
    circles = [make_circle(1000.0 + 500.0 * i, x=float(x), y=float(y)) for i, (x, y) in enumerate(positions)]
    beatmap = build_beatmap(circles, timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "style/grid snap" in tags


def test_playfield_constraint_when_objects_clustered() -> None:
    tp = make_timing_point(bpm=180.0)
    circles = [make_circle(1000.0 + 500.0 * i, x=100.0 + (i % 3) * 3, y=100.0 + (i % 4) * 3) for i in range(60)]
    beatmap = build_beatmap(circles, timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "gimmick/playfield constraint" in tags


def test_perfect_stacks_when_same_position_repeated() -> None:
    tp = make_timing_point(bpm=180.0)
    circles = [make_circle(1000.0 + 500.0 * i, x=256, y=192) for i in range(220)]
    beatmap = build_beatmap(circles, timing_points=[tp])

    tags = set(classify(beatmap).tags)
    assert "reading/perfect stacks" in tags
