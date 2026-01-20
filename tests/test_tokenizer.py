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