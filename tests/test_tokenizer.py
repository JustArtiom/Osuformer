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

tokenizer = Tokenizer(config=TokenizerConfig(), with_styles=False)

def dt_tokens(delta_ms: float):
  return [tokenizer.id_to_token[t] for t in tokenizer.encode_delta_time(delta_ms)]

def test_encoding_bpm_timing_point():
  beatmap = Beatmap()
  tp = TimingPoint(time=0, beat_length=500, uninherited=1)
  beatmap.timing_points.append(tp)

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  # No TS tokens emitted (those are inserted by TokenWindowBuilder)
  assert readable == ["SR_0", "MAP_START", "BPM_120", "MAP_END", "EOS"]

def test_encoding_sv_timing_point():
  beatmap = Beatmap(difficulty=Difficulty(slider_multiplier=2.0))
  tp = TimingPoint(time=0, beat_length=-50, uninherited=0)
  beatmap.timing_points.append(tp)

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  assert readable == ["SR_0", "MAP_START", "MAP_END", "EOS"]

def test_encoding_circle_hit_object():
  beatmap = Beatmap()
  circle = Circle(x=256, y=192, time=1000)
  beatmap.hit_objects.append(circle)

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  # No DT tokens between objects — only structural tokens
  assert readable == [
    "SR_0", "MAP_START",
    "OBJ_START", "T_CIRCLE", "X_16", "Y_12", "OBJ_END",
    "MAP_END", "EOS"
  ]

def test_encoding_circle_times_correct():
  beatmap = Beatmap()
  circle = Circle(x=256, y=192, time=1000)
  beatmap.hit_objects.append(circle)

  tokens, times, snaps = tokenizer.encode(beatmap)

  # All object tokens should have time=1000
  for i, readable in enumerate(tokenizer.id_to_token[t] for t in tokens):
    if readable in ("OBJ_START", "T_CIRCLE", "X_16", "Y_12", "OBJ_END"):
      assert times[i] == 1000.0

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

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  assert readable == [
    "SR_0", "MAP_START",
    "OBJ_START", "T_SLIDER",
    "X_16", "Y_12", "SV_1.4",
    "SL_240",
    "SLIDES_1",
    "SEG_BEZIER",
    "CP_0", "X_16", "Y_12",
    "CP_1", "X_19", "Y_12",
    "OBJ_END",
    "MAP_END", "EOS"
  ]

def test_encode_spinner_hit_object():
  """Spinners still use DT tokens for duration encoding."""
  beatmap = Beatmap()
  spinner = Spinner(x=256, y=192, time=3810, object_params=SpinnerObjectParams(end_time=5000))
  beatmap.hit_objects.append(spinner)

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  # Spinner uses DT for duration + SUSTAIN tokens
  # Duration = 1190ms, SUSTAIN_INTERVAL_MS = 100 -> 11 sustain tokens
  expected_dt = dt_tokens(1190)
  expected_sustain = ["SUSTAIN"] * 11  # 1190 / 100 = 11

  assert readable[:2] == ["SR_0", "MAP_START"]
  assert "OBJ_START" in readable
  assert "T_SPINNER" in readable

  # Find spinner section
  obj_start = readable.index("OBJ_START")
  obj_end = readable.index("OBJ_END")
  spinner_tokens = readable[obj_start:obj_end+1]

  assert spinner_tokens[0] == "OBJ_START"
  assert spinner_tokens[1] == "T_SPINNER"
  assert spinner_tokens[-1] == "OBJ_END"

  # Check DT tokens are inside the spinner
  dt_in_spinner = [t for t in spinner_tokens if t.startswith("DT_")]
  assert len(dt_in_spinner) > 0

  # Check SUSTAIN tokens are inside the spinner
  sustain_in_spinner = [t for t in spinner_tokens if t == "SUSTAIN"]
  assert len(sustain_in_spinner) == 11

def test_no_dt_between_objects():
  """Verify no DT tokens appear at the top level (between objects)."""
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(x=0, y=0, time=1000))
  beatmap.hit_objects.append(Circle(x=100, y=100, time=2000))

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  # DT tokens should NOT appear between objects
  in_object = False
  for tok in readable:
    if tok == "OBJ_START":
      in_object = True
    elif tok == "OBJ_END":
      in_object = False
    elif tok.startswith("DT_") and not in_object:
      assert False, f"Found DT token '{tok}' outside of an object"

def test_times_array_matches_objects():
  """The times array should correctly track absolute timestamps."""
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(x=0, y=0, time=1000))
  beatmap.hit_objects.append(Circle(x=100, y=100, time=2500))

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  # Find OBJ_START indices
  obj_starts = [i for i, t in enumerate(readable) if t == "OBJ_START"]
  assert len(obj_starts) == 2

  # First object at t=1000
  assert times[obj_starts[0]] == 1000.0
  # Second object at t=2500
  assert times[obj_starts[1]] == 2500.0

def test_snaps_array_matches_objects():
  """Snap computation should work for objects on beat."""
  beatmap = Beatmap()
  tp = TimingPoint(time=0, beat_length=500, uninherited=1)  # 120 BPM
  beatmap.timing_points.append(tp)
  # 500ms per beat, so t=500 is on beat (snap=1), t=250 is half beat (snap=2)
  beatmap.hit_objects.append(Circle(x=0, y=0, time=500))  # on beat
  beatmap.hit_objects.append(Circle(x=0, y=0, time=250))  # half beat

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  obj_starts = [i for i, t in enumerate(readable) if t == "OBJ_START"]
  # t=250 comes first (sorted by time)
  assert snaps[obj_starts[0]] == 2  # half beat snap
  assert snaps[obj_starts[1]] == 1  # on beat snap

def test_encode_returns_three_arrays():
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(x=0, y=0, time=0))

  result = tokenizer.encode(beatmap)
  assert len(result) == 3
  tokens, times, snaps = result
  assert len(tokens) == len(times) == len(snaps)

def test_empty_map_encoding():
  beatmap = Beatmap()
  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  assert readable == ["SR_0", "MAP_START", "MAP_END", "EOS"]

def test_timingpoint_priority_over_hitobject():
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(time=1000))
  beatmap.timing_points.append(TimingPoint(time=1000))

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  bpm_idx = next(i for i, tok in enumerate(readable) if tok.startswith("BPM_"))
  obj_idx = readable.index("OBJ_START")

  assert bpm_idx < obj_idx

def test_simultaneous_objects():
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(time=1000))
  beatmap.hit_objects.append(Circle(time=1000))

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  assert readable.count("OBJ_START") == 2
  assert readable.count("OBJ_END") == 2
  # No DT tokens at top level
  dt_outside = [t for t in readable if t.startswith("DT_")]
  assert len(dt_outside) == 0

def test_every_object_has_wrappers():
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(x=0, y=0, time=0))

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  assert readable.count("OBJ_START") == readable.count("OBJ_END")

def test_deterministic_encoding():
  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(x=10, y=10, time=100))

  assert tokenizer.encode(beatmap) == tokenizer.encode(beatmap)

def test_large_map():
  beatmap = Beatmap()
  for i in range(10000):
    beatmap.hit_objects.append(Circle(x=i%512, y=i%384, time=i*10))

  tokens, times, snaps = tokenizer.encode(beatmap)
  assert len(tokens) > 0
  assert len(tokens) == len(times) == len(snaps)

def test_decode_with_ts_tokens():
  """Test decoding a token sequence that includes TS/SNAP tokens (as inserted by window builder)."""
  test_tokens_with_ts = [
    'SR_0',
    'SPOS_50',
    'MAP_START',
    'TS_100', 'SNAP_4',
    'BPM_120',
    'TS_100', 'SNAP_4',
    'OBJ_START', 'T_CIRCLE', 'X_16', 'Y_12', 'OBJ_END',
    'TS_150', 'SNAP_2',
    'OBJ_START', 'T_CIRCLE', 'X_10', 'Y_8', 'OBJ_END',
    'MAP_END',
    'EOS'
  ]
  token_ids = [tokenizer.vocab[t] for t in test_tokens_with_ts]
  beatmap = tokenizer.decode(token_ids)

  # Should have 1 timing point and 2 circles
  assert len(beatmap.timing_points) == 1
  assert len(beatmap.hit_objects) == 2

  # First circle at TS_100 = 100 * 10ms = 1000ms
  assert beatmap.hit_objects[0].time == 1000.0
  # Second circle at TS_150 = 150 * 10ms = 1500ms
  assert beatmap.hit_objects[1].time == 1500.0

def test_sustain_tokens_in_slider():
  """Long sliders should get SUSTAIN tokens."""
  beatmap = Beatmap()
  beatmap.timing_points.append(TimingPoint(time=0, beat_length=500, uninherited=1))
  slider = Slider(
    x=256, y=192, time=0,
    object_params=SliderObjectParams(
      curves=[SliderCurve(curve_type=CurveType.LINEAR, curve_points=[(300,192)])],
      length=500,
      duration=1000,  # 1000ms duration -> 10 sustain tokens
    )
  )
  beatmap.hit_objects.append(slider)

  tokens, times, snaps = tokenizer.encode(beatmap)
  readable = [tokenizer.id_to_token[t] for t in tokens]

  sustain_count = readable.count("SUSTAIN")
  assert sustain_count == 10  # 1000ms / 100ms interval = 10


# Test tokens for grammar tests (new format without top-level DT, with TS/SNAP)
test_tokens = [
  'SR_2',
  'SPOS_0',
  'MAP_START',

  # BPM at TS_110 (= 1100ms)
  'TS_110', 'SNAP_0',
  'BPM_240',

  # Slider at TS_110
  'TS_110', 'SNAP_4',
  'OBJ_START', 'T_SLIDER', 'X_2', 'Y_21', 'SV_1.4', 'SL_140', 'SLIDES_1',
  'SEG_PERFECT', 'CP_0', 'X_5', 'Y_21', 'CP_1', 'X_10', 'Y_19', 'OBJ_END',

  # Circle at TS_180 (= 1800ms)
  'TS_180', 'SNAP_4',
  'OBJ_START', 'T_CIRCLE', 'X_12', 'Y_10', 'OBJ_END',

  # Slider at TS_193
  'TS_193', 'SNAP_4',
  'OBJ_START', 'T_SLIDER', 'X_12', 'Y_10', 'SV_1.4', 'SL_70', 'SLIDES_1',
  'SEG_LINEAR', 'CP_0', 'X_13', 'Y_16', 'OBJ_END',

  # Circle at TS_243
  'TS_243', 'SNAP_4',
  'OBJ_START', 'T_CIRCLE', 'X_5', 'Y_7', 'OBJ_END',

  # Spinner at TS_256
  'TS_256', 'SNAP_4',
  'OBJ_START', 'T_SPINNER', *['DT_10']*9, 'SUSTAIN', 'OBJ_END',

  'MAP_END',
  'EOS'
]
