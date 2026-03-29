from src.osu import HitSample, SampleSet


class TestHitSample:
    def test_parse_full(self):
        raw = "1:2:3:80:custom.wav"
        hs = HitSample(raw=raw)
        assert hs.normal_set == SampleSet.NORMAL
        assert hs.addition_set == SampleSet.SOFT
        assert hs.index == 3
        assert hs.volume == 80
        assert hs.filename == "custom.wav"
        assert str(hs) == raw

    def test_parse_default(self):
        raw = "0:0:0:0:"
        hs = HitSample(raw=raw)
        assert hs.normal_set == SampleSet.DEFAULT
        assert hs.addition_set == SampleSet.DEFAULT
        assert hs.index == 0
        assert hs.volume == 0
        assert hs.filename == ""
        assert str(hs) == raw

    def test_short_input_no_crash(self):
        hs = HitSample(raw="0:0")
        assert hs.normal_set == SampleSet.DEFAULT
        assert hs.addition_set == SampleSet.DEFAULT
        assert hs.index == 0
        assert hs.volume == 0

    def test_enum_types(self):
        hs = HitSample(raw="3:2:0:0:")
        assert hs.normal_set == SampleSet.DRUM
        assert hs.addition_set == SampleSet.SOFT
        assert isinstance(hs.normal_set, SampleSet)

    def test_constructor_kwargs(self):
        hs = HitSample(normal_set=SampleSet.SOFT, volume=70, filename="hit.wav")
        assert str(hs) == "2:0:0:70:hit.wav"
