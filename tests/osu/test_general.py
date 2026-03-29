from src.osu.sections import General


class TestGeneral:
    def test_parse_basic(self):
        g = General(raw="AudioFilename: song.mp3\nMode: 3\nStackLeniency: 0.5")
        assert g.audio_filename == "song.mp3"
        assert g.mode == 3
        assert g.stack_leniency == 0.5

    def test_sample_volume(self):
        g = General(raw="AudioFilename: a.mp3\nSampleVolume: 75")
        assert g.sample_volume == 75

    def test_str_omits_default_optionals(self):
        out = str(General(audio_filename="a.mp3"))
        assert "EpilepsyWarning" not in out
        assert "UseSkinSprites" not in out
        assert "SkinPreference" not in out

    def test_str_includes_optional_when_set(self):
        out = str(General(audio_filename="a.mp3", epilepsy_warning=1, widescreen_storyboard=1))
        assert "EpilepsyWarning: 1" in out
        assert "WidescreenStoryboard: 1" in out

    def test_stack_leniency_fmt(self):
        assert "StackLeniency: 0.7" in str(General(audio_filename="a.mp3", stack_leniency=0.7))

    def test_widescreen_storyboard_omitted_when_false(self):
        assert "WidescreenStoryboard" not in str(General(audio_filename="a.mp3", widescreen_storyboard=0))
