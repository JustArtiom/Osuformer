from src.osu import (
  TimingPoint,
  Circle,
  Slider,
  Spinner,
  HitSample,
  SliderObjectParams,
  SpinnerObjectParams,
  SliderCurve,
  CurveType,
  Beatmap
)

def test_uninherited_timing_point():
  raw="109,250,4,2,1,60,1,0"
  point = TimingPoint(raw=raw)
  assert point.time == 109.0
  assert point.beat_length == 250.0
  assert point.meter == 4
  assert point.sample_set == 2
  assert point.sample_index == 1
  assert point.volume == 60
  assert point.uninherited == 1
  assert point.effects == 0
  assert str(point) == raw

def test_inherited_timing_point():
  raw="109,-133.333333333333,4,2,1,60,0,0"
  point = TimingPoint(raw=raw)
  assert point.time == 109.0
  assert point.beat_length == -133.333333333333
  assert point.meter == 4
  assert point.sample_set == 2
  assert point.sample_index == 1
  assert point.volume == 60
  assert point.uninherited == 0
  assert point.effects == 0
  assert str(point) == raw

def test_timing_point_bpm():
  point = TimingPoint(beat_length=500)
  assert point.get_bpm() == 120.0

def test_timing_point_slider_velocity_multiplier():
  point = TimingPoint(beat_length=-50)
  assert point.get_slider_velocity_multiplier() == 2.0

def test_hit_object_type():
  raw_circle = "256,192,1000,1,0,0:0:0:0:"
  raw_slider = "256,192,2000,2,0,P|300:192,1,100,0:0|0:0,0:0:0:0:"
  raw_spinner = "256,192,3000,8,0,4000,0:0:0:0:"

  beatmap = Beatmap()
  assert beatmap.hit_object_type(raw_circle) == Circle
  assert beatmap.hit_object_type(raw_slider) == Slider
  assert beatmap.hit_object_type(raw_spinner) == Spinner

def test_hit_object_combo_flags():
  raw_new_combo = "256,192,1000,5,0,0:0:0:0:"
  raw_no_new_combo = "256,192,2000,1,0,0:0:0:0:"
  beatmap = Beatmap()
  type_new_combo = beatmap.hit_object_type(raw_new_combo)
  type_no_new_combo = beatmap.hit_object_type(raw_no_new_combo)

  obj_new_combo = type_new_combo(raw=raw_new_combo)
  obj_no_new_combo = type_no_new_combo(raw=raw_no_new_combo)

  assert obj_new_combo.is_new_combo()
  assert not obj_no_new_combo.is_new_combo()

def test_hit_object_combo_skip_count():
  raw_skip_2 = "256,192,1000,33,0,0:0:0:0:"
  raw_skip_0 = "256,192,2000,1,0,0:0:0:0:"
  beatmap = Beatmap()
  type_skip_2 = beatmap.hit_object_type(raw_skip_2)
  type_skip_0 = beatmap.hit_object_type(raw_skip_0)

  obj_skip_2 = type_skip_2(raw=raw_skip_2)
  obj_skip_0 = type_skip_0(raw=raw_skip_0)

  assert obj_skip_2.get_combo_skip_count() == 2
  assert obj_skip_0.get_combo_skip_count() == 0

def test_hit_sample():
  raw = "1:2:0:0:hit.wav"
  sample = HitSample(raw=raw)
  assert sample.normal_set == 1
  assert sample.addition_set == 2
  assert sample.index == 0
  assert sample.volume == 0
  assert sample.filename == "hit.wav"
  assert str(sample) == raw

def test_circle():
  raw = "268,64,3942,1,8,1:2:0:0:hit.wav"
  circle = Circle(raw=raw)
  assert circle.x == 268.0
  assert circle.y == 64.0
  assert circle.time == 3942.0
  assert circle.type == 1
  assert circle.hit_sound == 8
  assert str(circle) == raw
  print(circle.__dict__)

def test_slider_curve():
  raw = "B|128:64|192:128|256:64"
  curve = SliderCurve(raw=raw)
  assert curve.curve_type == CurveType.BEZIER
  assert curve.curve_points == [(128.0, 64.0), (192.0, 128.0), (256.0, 64.0)]
  assert str(curve) == raw

def test_slider_object_params():
  raw = "P|84:343|159:311,1,135.000005149842,6|2,1:2|0:0"
  params = SliderObjectParams(raw=raw)
  assert len(params.curves) == 1
  assert params.slides == 1
  assert params.length == 135.000005149842
  assert params.edge_sounds == [6, 2]
  assert params.edge_sets == [(1, 2), (0, 0)]
  assert str(params) == raw

def test_slider():
  raw = "197,23,1609,2,0,P|233:24|279:46,1,67.5000025749208,2|2,0:0|0:0,0:0:0:0:"
  slider = Slider(raw=raw)
  assert slider.x == 197.0
  assert slider.y == 23.0
  assert slider.time == 1609.0
  assert slider.type == 2
  assert slider.hit_sound == 0
  assert str(slider) == raw

def test_spinner_object_params():
  raw = "5000"
  params = SpinnerObjectParams(raw=raw)
  assert params.end_time == 5000.0
  assert str(params) == raw

def test_spinner():
  raw = "256,192,83979,8,0,84631,2:0:0:0:"
  spinner = Spinner(raw=raw)
  assert spinner.x == 256.0
  assert spinner.y == 192.0
  assert spinner.time == 83979.0
  assert spinner.type == 8
  assert spinner.object_params.end_time == 84631.0
  assert spinner.hit_sound == 0
  assert str(spinner) == raw