from src.osu import Circle, HitSound, SampleSet


class TestCircle:
    def test_parse_roundtrip(self):
        raw = "268,64,3942,1,8,1:2:0:0:hit.wav"
        c = Circle(raw=raw)
        assert c.x == 268.0
        assert c.y == 64.0
        assert c.time == 3942.0
        assert c.type == 1
        assert c.hit_sound == HitSound.CLAP
        assert str(c) == raw

    def test_no_hit_sample(self):
        c = Circle(raw="256,192,1000,1,2")
        assert c.hit_sound == HitSound.WHISTLE
        assert c.hit_sample.normal_set == SampleSet.DEFAULT

    def test_is_new_combo_true(self):
        assert Circle(raw="256,192,1000,5,0,0:0:0:0:").is_new_combo() is True

    def test_is_new_combo_false(self):
        assert Circle(raw="256,192,1000,1,0,0:0:0:0:").is_new_combo() is False

    def test_is_new_combo_type_override_zero(self):
        c = Circle(raw="256,192,1000,5,0,0:0:0:0:")
        assert c.is_new_combo(type_override=0) is False

    def test_is_new_combo_type_override(self):
        c = Circle()
        assert c.is_new_combo(type_override=4) is True
        assert c.is_new_combo(type_override=1) is False

    def test_combo_skip_count(self):
        # type=33 = 1 (circle) | 4 (new combo) | 32 (skip 2)
        assert Circle(raw="256,192,1000,33,0,0:0:0:0:").get_combo_skip_count() == 2

    def test_combo_skip_count_zero(self):
        assert Circle(raw="256,192,1000,5,0,0:0:0:0:").get_combo_skip_count() == 0

    def test_hit_sound_combined_flags(self):
        c = Circle(raw="256,192,1000,1,10,0:0:0:0:")
        assert HitSound.WHISTLE in c.hit_sound
        assert HitSound.CLAP in c.hit_sound
        assert HitSound.FINISH not in c.hit_sound
