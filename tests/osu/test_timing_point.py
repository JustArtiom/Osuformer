import pytest
from src.osu import TimingPoint, SampleSet, Effects


class TestTimingPoint:
    def test_parse_uninherited_roundtrip(self):
        raw = "109,250,4,2,1,60,1,0"
        tp = TimingPoint(raw=raw)
        assert tp.time == 109.0
        assert tp.beat_length == 250.0
        assert tp.meter == 4
        assert tp.sample_set == SampleSet.SOFT
        assert tp.sample_index == 1
        assert tp.volume == 60
        assert tp.uninherited == 1
        assert tp.effects == Effects.NONE
        assert str(tp) == raw

    def test_parse_inherited_roundtrip(self):
        raw = "5000,-133.333333333333,4,2,1,60,0,0"
        tp = TimingPoint(raw=raw)
        assert tp.uninherited == 0
        assert tp.beat_length == -133.333333333333
        assert str(tp) == raw

    def test_is_uninherited_property(self):
        assert TimingPoint(raw="0,500,4,1,0,100,1,0").is_uninherited is True
        assert TimingPoint(raw="0,-100,4,1,0,100,0,0").is_uninherited is False

    def test_is_kiai(self):
        assert TimingPoint(raw="0,500,4,1,0,100,1,1").is_kiai is True
        assert TimingPoint(raw="0,500,4,1,0,100,1,0").is_kiai is False

    def test_is_omit_first_bar_line(self):
        assert TimingPoint(raw="0,500,4,1,0,100,1,8").is_omit_first_bar_line is True
        assert TimingPoint(raw="0,500,4,1,0,100,1,0").is_omit_first_bar_line is False

    def test_kiai_and_omit_combined(self):
        tp = TimingPoint(raw="0,500,4,1,0,100,1,9")
        assert tp.is_kiai is True
        assert tp.is_omit_first_bar_line is True

    def test_get_bpm(self):
        assert TimingPoint(raw="0,500,4,1,0,100,1,0").get_bpm() == 120.0

    def test_get_bpm_raises_on_inherited(self):
        with pytest.raises(ValueError):
            TimingPoint(raw="0,-100,4,1,0,100,0,0").get_bpm()

    def test_get_slider_velocity_multiplier_uninherited(self):
        assert TimingPoint(beat_length=500.0, uninherited=1).get_slider_velocity_multiplier() == 1.0

    def test_get_slider_velocity_multiplier_2x(self):
        assert TimingPoint(raw="0,-50,4,1,0,100,0,0").get_slider_velocity_multiplier() == 2.0

    def test_get_slider_velocity_multiplier_default(self):
        assert TimingPoint(raw="0,-100,4,1,0,100,0,0").get_slider_velocity_multiplier() == 1.0

    def test_time_no_scientific_notation(self):
        tp = TimingPoint(time=1000000.0, beat_length=500.0, uninherited=1)
        assert "e" not in str(tp).lower()
        assert str(tp).startswith("1000000,")

    def test_sample_set_enum(self):
        assert TimingPoint(raw="0,500,4,3,0,100,1,0").sample_set == SampleSet.DRUM

    def test_effects_enum(self):
        tp = TimingPoint(raw="0,500,4,1,0,100,1,1")
        assert tp.effects == Effects.KIAI
        assert isinstance(tp.effects, Effects)

    def test_defaults(self):
        tp = TimingPoint()
        assert tp.meter == 4
        assert tp.volume == 100
        assert tp.uninherited == 1
        assert tp.effects == Effects.NONE
