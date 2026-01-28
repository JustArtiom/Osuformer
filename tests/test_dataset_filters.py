from src.config import ExperimentConfig
from src.data.dataset import is_map_valid
from src.osu import Beatmap, TimingPoint, Slider, SliderCurve, SliderObjectParams, CurveType


def test_is_map_valid_rejects_sliders_over_cp_limit():
    config = ExperimentConfig()
    config.tokenizer.SLIDER_CP_LIMIT = 2

    slider = Slider(
        x=256,
        y=192,
        time=0,
        object_params=SliderObjectParams(
            curves=[
                SliderCurve(
                    curve_type=CurveType.BEZIER,
                    curve_points=[(0.0, 0.0), (64.0, 64.0), (128.0, 128.0)],
                )
            ],
            slides=1,
            length=100.0,
        ),
    )

    beatmap = Beatmap(
        timing_points=[TimingPoint(time=0, beat_length=500, uninherited=1)],
        hit_objects=[slider],
        with_styles=False,
    )

    assert is_map_valid(beatmap, config) is False


def test_is_map_valid_rejects_sliders_over_slide_limit():
    config = ExperimentConfig()
    config.tokenizer.SLIDES_MAX = 10

    slider = Slider(
        x=256,
        y=192,
        time=0,
        object_params=SliderObjectParams(
            curves=[
                SliderCurve(
                    curve_type=CurveType.BEZIER,
                    curve_points=[(0.0, 0.0), (64.0, 64.0)],
                )
            ],
            slides=11,
            length=100.0,
        ),
    )

    beatmap = Beatmap(
        timing_points=[TimingPoint(time=0, beat_length=500, uninherited=1)],
        hit_objects=[slider],
        with_styles=False,
    )

    assert is_map_valid(beatmap, config) is False
