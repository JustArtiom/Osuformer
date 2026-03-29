from src.osu import Beatmap, Circle, Slider, Spinner, HoldNote

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

MANIA_BEATMAP = """\
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

V7_BEATMAP = """\
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


class TestBeatmap:
    def test_format_version(self):
        assert Beatmap(raw=FULL_BEATMAP, with_styles=False).format_version == 14

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

    def test_hit_object_types(self):
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
        assert bm.get_slider_velocity_multiplier_at(0) == 1.0
        assert bm.get_slider_velocity_multiplier_at(6000) == 1.0

    def test_slider_duration_calculated(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        slider = bm.hit_objects[2]
        assert isinstance(slider, Slider)
        assert slider.object_params.duration > 0

    def test_new_combo_on_second_circle(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert bm.hit_objects[1].is_new_combo() is True

    def test_v7_ar_defaults_to_od(self):
        bm = Beatmap(raw=V7_BEATMAP, with_styles=False)
        assert bm.format_version == 7
        assert bm.difficulty.approach_rate == bm.difficulty.overall_difficulty

    def test_v14_ar_not_overridden(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        assert bm.difficulty.approach_rate == 9.0

    def test_default_beatmap_sections_initialized(self):
        bm = Beatmap(with_styles=False)
        assert bm.general is not None
        assert bm.editor is not None
        assert bm.metadata is not None
        assert bm.difficulty is not None
        assert bm.events is not None
        assert bm.colours is not None

    def test_str_contains_version(self):
        assert str(Beatmap(raw=FULL_BEATMAP, with_styles=False)).startswith("osu file format v14")

    def test_str_no_triple_newline_before_colours(self):
        assert "\n\n\n[Colours]" not in str(Beatmap(raw=FULL_BEATMAP, with_styles=False))

    def test_hold_note_parsed(self):
        bm = Beatmap(raw=MANIA_BEATMAP, with_styles=False)
        assert len(bm.hit_objects) == 1
        assert isinstance(bm.hit_objects[0], HoldNote)
        assert bm.hit_objects[0].object_params.end_time == 2000.0

    def test_utf8_bom_handled(self):
        bm = Beatmap(raw="\ufeff" + FULL_BEATMAP, with_styles=False)
        assert bm.format_version == 14

    def test_get_previous_timing_point_alias(self):
        bm = Beatmap(raw=FULL_BEATMAP, with_styles=False)
        tp = bm.get_previous_timing_point(3000)
        assert tp is not None
        assert tp.time == 0
