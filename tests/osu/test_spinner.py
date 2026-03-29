from src.osu import Spinner, HitSound, SampleSet


class TestSpinner:
    def test_parse_roundtrip(self):
        raw = "256,192,83979,8,0,84631,2:0:0:0:"
        sp = Spinner(raw=raw)
        assert sp.x == 256.0
        assert sp.y == 192.0
        assert sp.time == 83979.0
        assert sp.type == 8
        assert sp.object_params.end_time == 84631.0
        assert sp.hit_sound == HitSound.NONE
        assert str(sp) == raw

    def test_parse_no_hit_sample(self):
        sp = Spinner(raw="256,192,5000,8,0,6000")
        assert sp.object_params.end_time == 6000.0
        assert sp.hit_sample.normal_set == SampleSet.DEFAULT

    def test_new_combo(self):
        assert Spinner(raw="256,192,5000,12,0,6000,0:0:0:0:").is_new_combo() is True
