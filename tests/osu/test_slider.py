import pytest
from src.osu import Slider, SliderCurve, SliderObjectParams, CurveType, HitSound, SampleSet


class TestSliderCurve:
    def test_bezier(self):
        raw = "B|128:64|192:128|256:64"
        sc = SliderCurve(raw=raw)
        assert sc.curve_type == CurveType.BEZIER
        assert sc.degree is None
        assert sc.curve_points == [(128.0, 64.0), (192.0, 128.0), (256.0, 64.0)]
        assert str(sc) == raw

    def test_linear(self):
        raw = "L|100:200|300:400"
        sc = SliderCurve(raw=raw)
        assert sc.curve_type == CurveType.LINEAR
        assert str(sc) == raw

    def test_perfect(self):
        raw = "P|150:200|250:150"
        sc = SliderCurve(raw=raw)
        assert sc.curve_type == CurveType.PERFECT
        assert str(sc) == raw

    def test_catmull(self):
        raw = "C|100:100|200:200|300:100"
        sc = SliderCurve(raw=raw)
        assert sc.curve_type == CurveType.CATMULL
        assert str(sc) == raw

    def test_bspline_with_degree(self):
        raw = "B3|100:200|200:300|300:200|400:300"
        sc = SliderCurve(raw=raw)
        assert sc.curve_type == CurveType.BEZIER
        assert sc.degree == 3
        assert len(sc.curve_points) == 4
        assert str(sc) == raw

    def test_bspline_degree_5(self):
        raw = "B5|50:50|100:100|150:50|200:100|250:50|300:100"
        sc = SliderCurve(raw=raw)
        assert sc.degree == 5
        assert str(sc) == raw

    def test_no_points(self):
        assert str(SliderCurve(curve_type=CurveType.BEZIER)) == "B"


class TestSliderObjectParams:
    def test_parse_basic_roundtrip(self):
        raw = "P|84:343|159:311,1,135.000005149842,6|2,1:2|0:0"
        params = SliderObjectParams(raw=raw)
        assert len(params.curves) == 1
        assert params.curves[0].curve_type == CurveType.PERFECT
        assert params.slides == 1
        assert params.length == pytest.approx(135.000005149842)
        assert params.edge_sounds == [HitSound(6), HitSound(2)]
        assert params.edge_sets == [(1, 2), (0, 0)]
        assert str(params) == raw

    def test_multi_curve_slider(self):
        raw = "B|100:200|200:200|B|200:200|300:100,1,200,0|0,0:0|0:0"
        params = SliderObjectParams(raw=raw)
        assert len(params.curves) == 2
        assert all(c.curve_type == CurveType.BEZIER for c in params.curves)

    def test_default_edge_sounds_sized_to_slides_plus_one(self):
        params = SliderObjectParams(slides=3)
        assert len(params.edge_sounds) == 4
        assert len(params.edge_sets) == 4

    def test_default_edge_sounds_slides_1(self):
        params = SliderObjectParams(slides=1)
        assert len(params.edge_sounds) == 2
        assert len(params.edge_sets) == 2

    def test_no_edge_sounds_in_raw(self):
        params = SliderObjectParams(raw="L|200:100|300:100,2,150")
        assert params.slides == 2
        assert len(params.edge_sounds) == 3
        assert len(params.edge_sets) == 3

    def test_bspline_degree_roundtrip(self):
        raw = "B3|100:200|200:300|300:200,1,150,0|0,0:0|0:0"
        params = SliderObjectParams(raw=raw)
        assert params.curves[0].degree == 3
        assert str(params) == raw


class TestSlider:
    def test_parse_roundtrip(self):
        raw = "197,23,1609,2,0,P|233:24|279:46,1,67.5000025749208,2|2,0:0|0:0,0:0:0:0:"
        s = Slider(raw=raw)
        assert s.x == 197.0
        assert s.y == 23.0
        assert s.time == 1609.0
        assert s.type == 2
        assert s.hit_sound == HitSound.NONE
        assert str(s) == raw

    def test_parse_no_hit_sample(self):
        s = Slider(raw="256,192,1000,2,0,L|356:192,1,100")
        assert s.hit_sample.normal_set == SampleSet.DEFAULT
        assert s.object_params.length == 100.0

    def test_edge_sounds_and_sets(self):
        s = Slider(raw="100,200,3000,2,0,B|200:200|300:100,1,200,2|0,1:2|0:0,0:0:0:0:")
        assert s.object_params.edge_sounds == [HitSound.WHISTLE, HitSound.NONE]
        assert s.object_params.edge_sets == [(1, 2), (0, 0)]

    def test_new_combo(self):
        assert Slider(raw="256,192,1000,6,0,L|356:192,1,100,0|0,0:0|0:0,0:0:0:0:").is_new_combo() is True
