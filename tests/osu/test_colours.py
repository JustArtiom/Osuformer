from src.osu.sections import Colours


class TestColours:
    def test_combo_colours(self):
        c = Colours(raw="Combo1 : 255,0,0\nCombo2 : 0,255,0\nCombo3 : 0,0,255")
        assert c.combo_colours[1] == (255, 0, 0)
        assert c.combo_colours[2] == (0, 255, 0)
        assert c.combo_colours[3] == (0, 0, 255)

    def test_slider_border(self):
        c = Colours(raw="SliderBorder : 255,255,255")
        assert c.slider_border == (255, 255, 255)

    def test_slider_track_override(self):
        c = Colours(raw="SliderTrackOverride : 100,200,50")
        assert c.slider_track_override == (100, 200, 50)

    def test_rgba_colour(self):
        c = Colours(raw="Combo1 : 255,0,0,128")
        assert c.combo_colours[1] == (255, 0, 0, 128)

    def test_str_roundtrip(self):
        c = Colours(raw="Combo1 : 255,0,0\nSliderBorder : 255,255,255")
        out = str(c)
        assert "Combo1 : 255,0,0" in out
        assert "SliderBorder : 255,255,255" in out

    def test_empty(self):
        assert str(Colours()) == ""
