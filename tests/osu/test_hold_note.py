from src.osu import HoldNote, SampleSet


class TestHoldNote:
    def test_parse_roundtrip(self):
        raw = "480,192,5000,128,0,6000:0:1:0:70:"
        hn = HoldNote(raw=raw)
        assert hn.x == 480.0
        assert hn.time == 5000.0
        assert hn.type == 128
        assert hn.object_params.end_time == 6000.0
        assert hn.hit_sample.addition_set == SampleSet.NORMAL
        assert hn.hit_sample.volume == 70
        assert str(hn) == raw

    def test_parse_default_hit_sample(self):
        hn = HoldNote(raw="256,192,1000,128,0,2000:0:0:0:0:")
        assert hn.object_params.end_time == 2000.0
        assert hn.hit_sample.normal_set == SampleSet.DEFAULT

    def test_new_combo(self):
        assert HoldNote(raw="480,192,1000,132,0,2000:0:0:0:0:").is_new_combo() is True
