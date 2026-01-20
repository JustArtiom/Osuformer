from src.osu import (
  Beatmap,
  TimingPoint,
  Difficulty,
  Circle,
  Slider,
  SliderObjectParams,
  SliderCurve,
  CurveType,
  Spinner,
  SpinnerObjectParams
)
from src.config import TokenizerConfig
from src.tokenizer import Tokenizer

def test_encoding_bpm_timing_point():
  beatmap = Beatmap()
  tp = TimingPoint(time=0, beat_length=500, uninherited=1)
  beatmap.timing_points.append(tp)

  tokenizer = Tokenizer(config=TokenizerConfig())
  tokens = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  assert readable == ["SR_0", "MAP_START", "DT_0", "TP_START", "BPM_120", "TP_END", "MAP_END", "EOS"]

def test_encoding_sv_timing_point():
  beatmap = Beatmap(difficulty=Difficulty(slider_multiplier=2.0))
  tp = TimingPoint(time=0, beat_length=-50, uninherited=0)
  beatmap.timing_points.append(tp)

  tokenizer = Tokenizer(config=TokenizerConfig())
  tokens = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  assert readable == ["SR_0", "MAP_START", "DT_0", "TP_START", "SV_4.0", "TP_END", "MAP_END", "EOS"]

def test_encoding_circle_hit_object():
  beatmap = Beatmap()
  circle = Circle(x=256, y=192, time=1000)
  beatmap.hit_objects.append(circle)

  tokenizer = Tokenizer(config=TokenizerConfig())
  tokens = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  assert readable == [
    "SR_0", "MAP_START", "DT_1000",
    "OBJ_START", "T_CIRCLE", "X_16", "Y_12", "OBJ_END",
    "MAP_END", "EOS"
  ]

def test_encoding_slider_hit_object():
  beatmap = Beatmap()
  slider = Slider(
    x=256, 
    y=192, 
    time=2000, 
    object_params=SliderObjectParams(
      curves=[SliderCurve(curve_type=CurveType.BEZIER, curve_points=[(256,192), (300,192)])],
      length=240
    )
  )
  beatmap.hit_objects.append(slider)

  tokenizer = Tokenizer(config=TokenizerConfig())
  tokens = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  assert readable == [
    "SR_0", "MAP_START", 
    "DT_1000", "DT_1000",
    "OBJ_START", "T_SLIDER", 
    "X_16", "Y_12",
    "SL_24",
    "SEG_BEZIER", 
    "CP_0", "X_16", "Y_12", 
    "CP_1", "X_19", "Y_12", 
    "OBJ_END",
    "MAP_END", "EOS"
  ]

def test_encode_spinner_hit_object():
  beatmap = Beatmap()
  spinner = Spinner(x=256, y=192, time=3810, object_params=SpinnerObjectParams(end_time=5000))
  beatmap.hit_objects.append(spinner)

  tokenizer = Tokenizer(config=TokenizerConfig())
  tokens = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  assert readable == [
    "SR_0", "MAP_START", 
    *["DT_1000"] * 3,
    *["DT_100"] * 8,
    "DT_10",
    "OBJ_START", "T_SPINNER", 
    "DT_1000", "DT_100", *["DT_10"]*9,
    "OBJ_END",
    "MAP_END", "EOS"
  ]

def test_encode_timing_margin_errors():
  beatmap = Beatmap()
  circle = Circle(x=256, y=192)
  beatmap.hit_objects.append(circle)
  tokenizer = Tokenizer(config=TokenizerConfig())

  tests = [1213, 1207]

  for test in tests:
    circle.time = test

    tokens = tokenizer.encode(beatmap)
    readable = [tokenizer.id_to_token[t] for t in tokens]

    assert readable == [
      "SR_0", "MAP_START", 
      "DT_1000", "DT_100", "DT_100", "DT_10",
      "OBJ_START", "T_CIRCLE", "X_16", "Y_12", "OBJ_END",
      "MAP_END", "EOS"
    ]

def test_encode_encode_margin_errors():
  beatmap = Beatmap()
  circle = Circle(x=256, y=192, time=0)
  beatmap.hit_objects.append(circle)
  tokenizer = Tokenizer(config=TokenizerConfig())

  def round_coord(value: float, axis) -> int:
    if axis == 'Y':
      return round(value / (384 / tokenizer.config.Y_BINS))
    elif axis == 'X':
      return round(value / (512 / tokenizer.config.X_BINS))

  tests = [(256, 192), (255, 191), (257, 193)]
  for test in tests:
    circle.x = test[0]
    circle.y = test[1]

    tokens = tokenizer.encode(beatmap)
    readable = [tokenizer.id_to_token[t] for t in tokens]

    assert readable == [
      "SR_0", "MAP_START", 
      "DT_0",
      "OBJ_START", "T_CIRCLE", 
      f"X_16", f"Y_12", 
      "OBJ_END",
      "MAP_END", "EOS"
    ]

def test_encoding_slider_length_errors():
  beatmap = Beatmap()
  slider = Slider(
    x=256, 
    y=192, 
    time=2000, 
    object_params=SliderObjectParams(
      curves=[SliderCurve(curve_type=CurveType.BEZIER, curve_points=[(256,192), (300,192)])],
      length=240
    )
  )
  beatmap.hit_objects.append(slider)

  tokenizer = Tokenizer(config=TokenizerConfig())

  tests = [238, 242]

  for test in tests:
    slider.object_params.length = test

    tokens = tokenizer.encode(beatmap)
    readable = [tokenizer.id_to_token[t] for t in tokens]

    assert readable == [
      "SR_0", "MAP_START", 
      "DT_1000", "DT_1000",
      "OBJ_START", "T_SLIDER", 
      "X_16", "Y_12",
      "SL_24",
      "SEG_BEZIER", 
      "CP_0", "X_16", "Y_12", 
      "CP_1", "X_19", "Y_12", 
      "OBJ_END",
      "MAP_END", "EOS"
    ]

def test_timingpoint_priority_over_hitobject():
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(time=1000))
  beatmap.timing_points.append(TimingPoint(time=1000))

  tokens = Tokenizer(TokenizerConfig()).encode(beatmap)
  readable = [Tokenizer(TokenizerConfig()).id_to_token[t] for t in tokens]

  tp_idx = readable.index("TP_START")
  obj_idx = readable.index("OBJ_START")

  assert tp_idx < obj_idx

def test_empty_map_encoding():
  beatmap = Beatmap()
  tokens = Tokenizer(TokenizerConfig()).encode(beatmap)
  readable = [Tokenizer(TokenizerConfig()).id_to_token[t] for t in tokens]

  assert readable == ["SR_0","MAP_START","MAP_END","EOS"]

def test_simultaneous_objects():
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(time=1000))
  beatmap.hit_objects.append(Circle(time=1000))

  tokens = Tokenizer(TokenizerConfig()).encode(beatmap)
  readable = [Tokenizer(TokenizerConfig()).id_to_token[t] for t in tokens]

  assert readable.count("DT_0") == 1

def test_every_object_has_wrappers():
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(x=0,y=0,time=0))

  readable = [Tokenizer(TokenizerConfig()).id_to_token[t] for t in Tokenizer(TokenizerConfig()).encode(beatmap)]

  assert readable.count("OBJ_START") == readable.count("OBJ_END")

def test_deterministic_encoding():
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(x=10,y=10,time=100))

  tok = Tokenizer(TokenizerConfig())
  assert tok.encode(beatmap) == tok.encode(beatmap)

def test_large_map():
  beatmap = Beatmap()
  for i in range(10000):
    beatmap.hit_objects.append(Circle(x=i%512,y=i%384,time=i*10))

  tokens = Tokenizer(TokenizerConfig()).encode(beatmap)
  assert len(tokens) > 0