import pytest
from src.osu import (
    Beatmap,
    Circle,
    Slider,
    Spinner,
    HoldNote,
    HitSample,
    SliderCurve,
    SliderObjectParams,
    SpinnerObjectParams,
    HoldNoteObjectParams,
    TimingPoint,
    CurveType,
    SampleSet,
    HitSound,
    Effects,
)
from src.osu.sections import General, Difficulty, Editor, Metadata, Colours, Events
from src.osu.sections.events import BackgroundEvent, VideoEvent, BreakEvent


# ---------------------------------------------------------------------------
# TimingPoint
# ---------------------------------------------------------------------------

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
        tp_uninherited = TimingPoint(raw="0,500,4,1,0,100,1,0")
        tp_inherited = TimingPoint(raw="0,-100,4,1,0,100,0,0")
        assert tp_uninherited.is_uninherited is True
        assert tp_inherited.is_uninherited is False

    def test_is_kiai(self):
        tp_kiai = TimingPoint(raw="0,500,4,1,0,100,1,1")
        tp_no_kiai = TimingPoint(raw="0,500,4,1,0,100,1,0")
        assert tp_kiai.is_kiai is True
        assert tp_no_kiai.is_kiai is False

    def test_is_omit_first_bar_line(self):
        tp = TimingPoint(raw="0,500,4,1,0,100,1,8")
        assert tp.is_omit_first_bar_line is True
        tp2 = TimingPoint(raw="0,500,4,1,0,100,1,0")
        assert tp2.is_omit_first_bar_line is False

    def test_kiai_and_omit_combined(self):
        tp = TimingPoint(raw="0,500,4,1,0,100,1,9")  # 1 | 8 = 9
        assert tp.is_kiai is True
        assert tp.is_omit_first_bar_line is True

    def test_get_bpm(self):
        tp = TimingPoint(raw="0,500,4,1,0,100,1,0")
        assert tp.get_bpm() == 120.0

    def test_get_bpm_raises_on_inherited(self):
        tp = TimingPoint(raw="0,-100,4,1,0,100,0,0")
        with pytest.raises(ValueError):
            tp.get_bpm()

    def test_get_slider_velocity_multiplier_on_uninherited(self):
        tp = TimingPoint(beat_length=500.0, uninherited=1)
        assert tp.get_slider_velocity_multiplier() == 1.0

    def test_get_slider_velocity_multiplier_on_inherited(self):
        tp = TimingPoint(raw="0,-50,4,1,0,100,0,0")
        assert tp.get_slider_velocity_multiplier() == 2.0

    def test_get_slider_velocity_multiplier_default(self):
        tp = TimingPoint(raw="0,-100,4,1,0,100,0,0")
        assert tp.get_slider_velocity_multiplier() == 1.0

    def test_time_no_scientific_notation(self):
        # Times > 1,000,000ms (16+ min song) must not produce 1e+06
        tp = TimingPoint(time=1000000.0, beat_length=500.0, uninherited=1)
        assert "e" not in str(tp).lower()
        assert str(tp).startswith("1000000,")

    def test_sample_set_enum(self):
        tp = TimingPoint(raw="0,500,4,3,0,100,1,0")
        assert tp.sample_set == SampleSet.DRUM

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


# ---------------------------------------------------------------------------
# HitSample
# ---------------------------------------------------------------------------

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

    def test_sampleset_enum_types(self):
        hs = HitSample(raw="3:2:0:0:")
        assert hs.normal_set == SampleSet.DRUM
        assert hs.addition_set == SampleSet.SOFT
        assert isinstance(hs.normal_set, SampleSet)

    def test_constructor_kwargs(self):
        hs = HitSample(normal_set=SampleSet.SOFT, volume=70, filename="hit.wav")
        assert str(hs) == "2:0:0:70:hit.wav"


# ---------------------------------------------------------------------------
# Circle
# ---------------------------------------------------------------------------

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
        raw = "256,192,1000,1,2"
        c = Circle(raw=raw)
        assert c.x == 256.0
        assert c.hit_sound == HitSound.WHISTLE
        assert c.hit_sample.normal_set == SampleSet.DEFAULT

    def test_is_new_combo_true(self):
        c = Circle(raw="256,192,1000,5,0,0:0:0:0:")
        assert c.is_new_combo() is True

    def test_is_new_combo_false(self):
        c = Circle(raw="256,192,1000,1,0,0:0:0:0:")
        assert c.is_new_combo() is False

    def test_is_new_combo_type_override_zero_bug_fixed(self):
        # Old bug: `type or self.type` made type_override=0 use self.type instead
        c = Circle(raw="256,192,1000,5,0,0:0:0:0:")
        assert c.is_new_combo(type_override=0) is False  # must respect explicit 0

    def test_is_new_combo_type_override(self):
        c = Circle()
        assert c.is_new_combo(type_override=4) is True
        assert c.is_new_combo(type_override=1) is False

    def test_combo_skip_count(self):
        # type=33 = 1 (circle) | 4 (new combo) | 32 (skip 2: bits 4&5 = 0b010)
        c = Circle(raw="256,192,1000,33,0,0:0:0:0:")
        assert c.get_combo_skip_count() == 2

    def test_combo_skip_count_zero(self):
        c = Circle(raw="256,192,1000,5,0,0:0:0:0:")
        assert c.get_combo_skip_count() == 0

    def test_hit_sound_combined_flags(self):
        # hit_sound=10 = WHISTLE(2) | CLAP(8)
        c = Circle(raw="256,192,1000,1,10,0:0:0:0:")
        assert HitSound.WHISTLE in c.hit_sound
        assert HitSound.CLAP in c.hit_sound
        assert HitSound.FINISH not in c.hit_sound


# ---------------------------------------------------------------------------
# SliderCurve
# ---------------------------------------------------------------------------

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
        sc = SliderCurve(curve_type=CurveType.BEZIER)
        assert str(sc) == "B"


# ---------------------------------------------------------------------------
# SliderObjectParams
# ---------------------------------------------------------------------------

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
        # B-spline segment break: two B segments
        raw = "B|100:200|200:200|B|200:200|300:100,1,200,0|0,0:0|0:0"
        params = SliderObjectParams(raw=raw)
        assert len(params.curves) == 2
        assert params.curves[0].curve_type == CurveType.BEZIER
        assert params.curves[1].curve_type == CurveType.BEZIER

    def test_default_edge_sounds_sized_correctly(self):
        # slides=3 → edge_sounds should have 4 entries (slides+1)
        params = SliderObjectParams(slides=3)
        assert len(params.edge_sounds) == 4
        assert len(params.edge_sets) == 4

    def test_default_edge_sounds_slides_1(self):
        params = SliderObjectParams(slides=1)
        assert len(params.edge_sounds) == 2
        assert len(params.edge_sets) == 2

    def test_no_edge_sounds_in_raw(self):
        raw = "L|200:100|300:100,2,150"
        params = SliderObjectParams(raw=raw)
        assert params.slides == 2
        assert len(params.edge_sounds) == 3  # slides+1
        assert len(params.edge_sets) == 3

    def test_bspline_degree_in_curve(self):
        raw = "B3|100:200|200:300|300:200,1,150,0|0,0:0|0:0"
        params = SliderObjectParams(raw=raw)
        assert len(params.curves) == 1
        assert params.curves[0].degree == 3
        assert str(params) == raw


# ---------------------------------------------------------------------------
# Slider
# ---------------------------------------------------------------------------

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
        raw = "256,192,1000,2,0,L|356:192,1,100"
        s = Slider(raw=raw)
        assert s.hit_sample.normal_set == SampleSet.DEFAULT
        assert s.object_params.length == 100.0

    def test_edge_sounds_and_sets(self):
        raw = "100,200,3000,2,0,B|200:200|300:100,1,200,2|0,1:2|0:0,0:0:0:0:"
        s = Slider(raw=raw)
        assert s.object_params.edge_sounds == [HitSound.WHISTLE, HitSound.NONE]
        assert s.object_params.edge_sets == [(1, 2), (0, 0)]

    def test_new_combo_slider(self):
        raw = "256,192,1000,6,0,L|356:192,1,100,0|0,0:0|0:0,0:0:0:0:"
        s = Slider(raw=raw)
        assert s.is_new_combo() is True


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

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
        raw = "256,192,5000,8,0,6000"
        sp = Spinner(raw=raw)
        assert sp.object_params.end_time == 6000.0
        assert sp.hit_sample.normal_set == SampleSet.DEFAULT

    def test_new_combo_spinner(self):
        raw = "256,192,5000,12,0,6000,0:0:0:0:"
        sp = Spinner(raw=raw)
        assert sp.is_new_combo() is True


# ---------------------------------------------------------------------------
# HoldNote (mania)
# ---------------------------------------------------------------------------

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
        raw = "256,192,1000,128,0,2000:0:0:0:0:"
        hn = HoldNote(raw=raw)
        assert hn.object_params.end_time == 2000.0
        assert hn.hit_sample.normal_set == SampleSet.DEFAULT

    def test_new_combo_hold(self):
        raw = "480,192,1000,132,0,2000:0:0:0:0:"  # 128 | 4
        hn = HoldNote(raw=raw)
        assert hn.is_new_combo() is True


# ---------------------------------------------------------------------------
# Events section
# ---------------------------------------------------------------------------

class TestEvents:
    def test_background(self):
        raw = '0,0,"bg.jpg",0,0'
        ev = Events(raw=raw)
        assert ev.background is not None
        assert ev.background.filename == "bg.jpg"
        assert ev.background.x_offset == 0

    def test_background_with_offset(self):
        raw = '0,0,"bg.jpg",10,-20'
        ev = Events(raw=raw)
        assert ev.background is not None
        assert ev.background.x_offset == 10
        assert ev.background.y_offset == -20

    def test_video(self):
        raw = 'Video,0,"intro.avi"'
        ev = Events(raw=raw)
        assert ev.video is not None
        assert ev.video.filename == "intro.avi"
        assert ev.video.start_time == 0

    def test_video_numeric_type(self):
        raw = '1,500,"video.mp4"'
        ev = Events(raw=raw)
        assert ev.video is not None
        assert ev.video.filename == "video.mp4"
        assert ev.video.start_time == 500

    def test_breaks(self):
        raw = "2,10000,15000"
        ev = Events(raw=raw)
        assert len(ev.breaks) == 1
        assert ev.breaks[0].start_time == 10000.0
        assert ev.breaks[0].end_time == 15000.0

    def test_multiple_breaks(self):
        raw = "2,5000,8000\n2,20000,25000"
        ev = Events(raw=raw)
        assert len(ev.breaks) == 2

    def test_comments_skipped(self):
        raw = "//Background and Video events\n0,0,\"bg.jpg\",0,0"
        ev = Events(raw=raw)
        assert ev.background is not None

    def test_empty_raw(self):
        ev = Events()
        assert ev.background is None
        assert ev.video is None
        assert ev.breaks == []

    def test_background_roundtrip(self):
        raw_in = '0,0,"bg.jpg",0,0'
        ev = Events(raw=raw_in)
        out = str(ev)
        assert '"bg.jpg"' in out

    def test_break_roundtrip(self):
        ev = Events(raw="2,10000,15000")
        assert "2,10000,15000" in str(ev)


# ---------------------------------------------------------------------------
# Colours section
# ---------------------------------------------------------------------------

class TestColours:
    def test_combo_colours(self):
        raw = "Combo1 : 255,0,0\nCombo2 : 0,255,0\nCombo3 : 0,0,255"
        c = Colours(raw=raw)
        assert c.combo_colours[1] == (255, 0, 0)
        assert c.combo_colours[2] == (0, 255, 0)
        assert c.combo_colours[3] == (0, 0, 255)

    def test_slider_border(self):
        raw = "SliderBorder : 255,255,255"
        c = Colours(raw=raw)
        assert c.slider_border == (255, 255, 255)

    def test_slider_track_override(self):
        raw = "SliderTrackOverride : 100,200,50"
        c = Colours(raw=raw)
        assert c.slider_track_override == (100, 200, 50)

    def test_rgba_colour(self):
        raw = "Combo1 : 255,0,0,128"
        c = Colours(raw=raw)
        assert c.combo_colours[1] == (255, 0, 0, 128)

    def test_str_roundtrip(self):
        raw = "Combo1 : 255,0,0\nSliderBorder : 255,255,255"
        c = Colours(raw=raw)
        out = str(c)
        assert "Combo1 : 255,0,0" in out
        assert "SliderBorder : 255,255,255" in out

    def test_empty(self):
        c = Colours()
        assert str(c) == ""


# ---------------------------------------------------------------------------
# General section
# ---------------------------------------------------------------------------

class TestGeneral:
    def test_parse_basic(self):
        raw = "AudioFilename: song.mp3\nMode: 3\nStackLeniency: 0.5"
        g = General(raw=raw)
        assert g.audio_filename == "song.mp3"
        assert g.mode == 3
        assert g.stack_leniency == 0.5

    def test_sample_volume(self):
        raw = "AudioFilename: a.mp3\nSampleVolume: 75"
        g = General(raw=raw)
        assert g.sample_volume == 75

    def test_str_omits_default_optionals(self):
        g = General(audio_filename="a.mp3")
        out = str(g)
        assert "EpilepsyWarning" not in out
        assert "UseSkinSprites" not in out
        assert "SkinPreference" not in out

    def test_str_includes_optional_when_set(self):
        g = General(audio_filename="a.mp3", epilepsy_warning=1, widescreen_storyboard=1)
        out = str(g)
        assert "EpilepsyWarning: 1" in out
        assert "WidescreenStoryboard: 1" in out

    def test_stack_leniency_fmt(self):
        g = General(audio_filename="a.mp3", stack_leniency=0.7)
        out = str(g)
        assert "StackLeniency: 0.7" in out

    def test_widescreen_storyboard_not_emitted_when_false(self):
        g = General(audio_filename="a.mp3", widescreen_storyboard=0)
        assert "WidescreenStoryboard" not in str(g)


# ---------------------------------------------------------------------------
# Difficulty section
# ---------------------------------------------------------------------------

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
        d = Difficulty(raw=raw)
        assert d._has_approach_rate is True

    def test_has_approach_rate_flag_unset(self):
        raw = "HPDrainRate:5\nCircleSize:4\nOverallDifficulty:7\nSliderMultiplier:1.4\nSliderTickRate:1"
        d = Difficulty(raw=raw)
        assert d._has_approach_rate is False


# ---------------------------------------------------------------------------
# Beatmap
# ---------------------------------------------------------------------------

FULL_BEATMAP = """\
osu file format v14

[General]
AudioFilename: audio.mp3
AudioLeadIn: 0
PreviewTime: 5000
Countdown: 0
SampleSet: Soft
StackLeniency: 0.7
Mode: 0

[Editor]
DistanceSpacing: 1.2
BeatDivisor: 4
GridSize: 4
TimelineZoom: 1

[Metadata]
Title:Test Song
TitleUnicode:Test Song
Artist:Artist
ArtistUnicode:Artist
Creator:Mapper
Version:Hard
Source:
Tags:tag1 tag2
BeatmapID:123
BeatmapSetID:456

[Difficulty]
HPDrainRate:6
CircleSize:4
OverallDifficulty:8
ApproachRate:9
SliderMultiplier:1.8
SliderTickRate:1

[Events]
0,0,"bg.jpg",0,0
2,10000,15000

[TimingPoints]
0,500,4,1,0,100,1,0
5000,-100,4,2,0,80,0,1

[Colours]
Combo1 : 255,128,0
SliderBorder : 255,255,255

[HitObjects]
256,192,1000,1,2,0:0:0:0:
100,100,2000,5,0,0:0:0:0:
200,200,3000,2,0,B|300:200|400:100,1,200,2|0,1:0|0:0,0:0:0:0:
256,192,10000,12,0,11000,0:0:0:0:
"""


class TestBeatmap:
    def test_format_version(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert bm.format_version == 14

    def test_sections_parsed(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert bm.general.audio_filename == "audio.mp3"
        assert bm.metadata.title == "Test Song"
        assert bm.metadata.beatmap_id == 123
        assert bm.difficulty.approach_rate == 9.0
        assert bm.difficulty.slider_multiplier == 1.8

    def test_timing_points(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert len(bm.timing_points) == 2
        assert bm.timing_points[0].is_uninherited is True
        assert bm.timing_points[1].is_uninherited is False
        assert bm.timing_points[1].is_kiai is True

    def test_hit_objects_types(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert len(bm.hit_objects) == 4
        assert isinstance(bm.hit_objects[0], Circle)
        assert isinstance(bm.hit_objects[1], Circle)
        assert isinstance(bm.hit_objects[2], Slider)
        assert isinstance(bm.hit_objects[3], Spinner)

    def test_colours(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert bm.colours.combo_colours[1] == (255, 128, 0)
        assert bm.colours.slider_border == (255, 255, 255)

    def test_events(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert bm.events.background is not None
        assert bm.events.background.filename == "bg.jpg"
        assert len(bm.events.breaks) == 1
        assert bm.events.breaks[0].end_time == 15000.0

    def test_get_bpm_at(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert bm.get_bpm_at(0) == 120.0
        assert bm.get_bpm_at(3000) == 120.0

    def test_get_slider_velocity_multiplier_at(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert bm.get_slider_velocity_multiplier_at(0) == 1.0    # no inherited TP yet
        assert bm.get_slider_velocity_multiplier_at(6000) == 1.0  # beat_length=-100 → 1.0x

    def test_slider_duration_calculated(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        slider = bm.hit_objects[2]
        assert isinstance(slider, Slider)
        assert slider.object_params.duration > 0

    def test_new_combo_on_second_circle(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert bm.hit_objects[1].is_new_combo() is True

    def test_format_version_v7_ar_defaults_to_od(self):
        raw = """\
osu file format v7

[General]
AudioFilename: a.mp3
Mode: 0

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:7
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,500,4,1,0,100,1,0

[HitObjects]
256,192,1000,1,0,0:0:0:0:
"""
        bm = Beatmap(raw=raw, with_styles=False)
        assert bm.format_version == 7
        assert bm.difficulty.approach_rate == bm.difficulty.overall_difficulty

    def test_format_version_v14_ar_not_overridden(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert bm.difficulty.approach_rate == 9.0  # explicitly set, not overridden

    def test_default_beatmap_has_all_sections(self):
        bm = Beatmap(with_styles=False)
        assert bm.general is not None
        assert bm.editor is not None
        assert bm.metadata is not None
        assert bm.difficulty is not None
        assert bm.events is not None
        assert bm.colours is not None

    def test_str_contains_version(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert str(bm).startswith("osu file format v14")

    def test_str_no_extra_newlines_before_colours(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        out = str(bm)
        # Should not have triple newline (the old bug)
        assert "\n\n\n[Colours]" not in out

    def test_hold_note_parsed(self):
        raw = """\
osu file format v14

[General]
AudioFilename: a.mp3
Mode: 3

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:7
ApproachRate:7
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,500,4,1,0,100,1,0

[HitObjects]
256,192,1000,128,0,2000:0:0:0:0:
"""
        bm = Beatmap(raw=raw, with_styles=False)
        assert len(bm.hit_objects) == 1
        assert isinstance(bm.hit_objects[0], HoldNote)
        assert bm.hit_objects[0].object_params.end_time == 2000.0

    def test_utf8_bom_handled(self):
        bom_raw = "\ufeff" + FULL_BEATMAP
        bm = Beatmap(raw=bom_raw, with_styles=False)
        assert bm.format_version == 14

    def test_get_previous_timing_point_alias(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        tp = bm.get_previous_timing_point(3000)
        assert tp is not None
        assert tp.time == 0


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestFmt:
    def test_integer_float(self):
        from src.osu.utils import fmt
        assert fmt(100.0) == "100"
        assert fmt(0.0) == "0"
        assert fmt(-5.0) == "-5"

    def test_decimal(self):
        from src.osu.utils import fmt
        assert fmt(1.5) == "1.5"
        assert fmt(0.001) == "0.001"

    def test_no_scientific_notation(self):
        from src.osu.utils import fmt
        assert "e" not in fmt(1e-7).lower()
        assert fmt(1e-7) == "0.0000001"

    def test_negative_zero(self):
        from src.osu.utils import fmt
        assert fmt(-0.0) == "0"

    def test_nan_returns_zero(self):
        from src.osu.utils import fmt
        assert fmt(float("nan")) == "0"

    def test_inf_returns_zero(self):
        from src.osu.utils import fmt
        assert fmt(float("inf")) == "0"
        assert fmt(float("-inf")) == "0"
