from src.osu.sections import Difficulty


class TestDifficulty:
    def test_parse_roundtrip(self):
        raw = "HPDrainRate:7\nCircleSize:4\nOverallDifficulty:8\nApproachRate:9\nSliderMultiplier:1.8\nSliderTickRate:1"
        d = Difficulty(raw=raw)
        assert d.hp_drain_rate == 7.0
        assert d.circle_size == 4.0
        assert d.overall_difficulty == 8.0
        assert d.approach_rate == 9.0
        assert d.slider_multiplier == 1.8
        assert d.slider_tick_rate == 1.0
        assert str(d) == raw

    def test_has_approach_rate_flag_set(self):
        raw = "HPDrainRate:5\nCircleSize:4\nOverallDifficulty:7\nApproachRate:8\nSliderMultiplier:1.4\nSliderTickRate:1"
        assert Difficulty(raw=raw)._has_approach_rate is True

    def test_has_approach_rate_flag_unset(self):
        raw = "HPDrainRate:5\nCircleSize:4\nOverallDifficulty:7\nSliderMultiplier:1.4\nSliderTickRate:1"
        assert Difficulty(raw=raw)._has_approach_rate is False
