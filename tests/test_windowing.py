import numpy as np
from src.config import TokenizerConfig, ExperimentConfig
from src.tokenizer import Tokenizer
from src.data.dataset import TokenWindowBuilder
from src.osu import (
  Beatmap, Circle, TimingPoint, Slider, SliderObjectParams, SliderCurve, CurveType
)


def _make_builder(max_tokens=1024, overlap=0.5):
  config = TokenizerConfig()
  tokenizer = Tokenizer(config)
  builder = TokenWindowBuilder(
    tokenizer=tokenizer,
    max_tokens=max_tokens,
    overlap_ratio=overlap,
  )
  return builder, tokenizer


def test_window_inserts_ts_snap():
  """Builder correctly inserts TS/SNAP before each OBJ_START and BPM token."""
  builder, tokenizer = _make_builder()

  beatmap = Beatmap()
  beatmap.timing_points.append(TimingPoint(time=1000, beat_length=500, uninherited=1))
  beatmap.hit_objects.append(Circle(x=256, y=192, time=1000))
  beatmap.hit_objects.append(Circle(x=100, y=100, time=2000))

  tokens, times, snaps = tokenizer.encode(beatmap)

  window_tokens, loss_mask = builder.build(
    tokens=np.array(tokens),
    times=np.array(times, dtype=np.float64),
    snaps=np.array(snaps),
    audio_start_ms=0,
    audio_end_ms=8000,
  )

  readable = [tokenizer.id_to_token[t] for t in window_tokens]

  # TS and SNAP tokens should appear in the output
  ts_tokens = [t for t in readable if t.startswith("TS_")]
  snap_tokens = [t for t in readable if t.startswith("SNAP_")]
  assert len(ts_tokens) > 0, "No TS tokens inserted"
  assert len(snap_tokens) > 0, "No SNAP tokens inserted"

  # Each OBJ_START should be preceded by a SNAP token (which is preceded by a TS token)
  for i, tok in enumerate(readable):
    if tok == "OBJ_START" and i >= 2:
      assert readable[i-1].startswith("SNAP_"), \
        f"OBJ_START at index {i} not preceded by SNAP, got '{readable[i-1]}'"
      assert readable[i-2].startswith("TS_"), \
        f"OBJ_START at index {i} not preceded by TS/SNAP pair, got '{readable[i-2]}'/'{readable[i-1]}'"

  # BPM tokens should also be preceded by TS/SNAP
  for i, tok in enumerate(readable):
    if tok.startswith("BPM_") and i >= 2:
      assert readable[i-1].startswith("SNAP_"), \
        f"BPM at index {i} not preceded by SNAP"
      assert readable[i-2].startswith("TS_"), \
        f"BPM at index {i} not preceded by TS"


def test_window_spos_prefix():
  """SPOS token appears at the start of the window."""
  builder, tokenizer = _make_builder()

  beatmap = Beatmap()
  beatmap.hit_objects.append(Circle(x=0, y=0, time=5000))

  tokens, times, snaps = tokenizer.encode(beatmap)

  window_tokens, loss_mask = builder.build(
    tokens=np.array(tokens),
    times=np.array(times, dtype=np.float64),
    snaps=np.array(snaps),
    audio_start_ms=2000,
    audio_end_ms=10000,
  )

  readable = [tokenizer.id_to_token[t] for t in window_tokens]

  # SPOS should be in the prefix, before MAP_START
  map_start_idx = readable.index("MAP_START")
  spos_found = any(t.startswith("SPOS_") for t in readable[:map_start_idx])
  assert spos_found, f"No SPOS token found before MAP_START. Prefix: {readable[:map_start_idx+1]}"


def test_window_loss_mask():
  """Context region has loss_mask=0, target region has loss_mask=1."""
  builder, tokenizer = _make_builder(overlap=0.5)

  beatmap = Beatmap()
  # Object at 1000ms (context when window starts at 0, overlap=0.5 -> predict_start=4000)
  beatmap.hit_objects.append(Circle(x=0, y=0, time=1000))
  # Object at 5000ms (target - after predict_start_ms=4000)
  beatmap.hit_objects.append(Circle(x=100, y=100, time=5000))

  tokens, times, snaps = tokenizer.encode(beatmap)

  window_tokens, loss_mask = builder.build(
    tokens=np.array(tokens),
    times=np.array(times, dtype=np.float64),
    snaps=np.array(snaps),
    audio_start_ms=0,
    audio_end_ms=8000,
  )

  readable = [tokenizer.id_to_token[t] for t in window_tokens]

  # Loss mask should have some False (context/prefix) and some True (target)
  assert not loss_mask.all(), "Loss mask should not be all True (should have context)"
  assert loss_mask.any(), "Loss mask should have some True values (target region)"

  # Prefix tokens (SR, SPOS, MAP_START) should have loss_mask=0
  map_start_indices = [i for i, t in enumerate(readable) if t == "MAP_START"]
  if map_start_indices:
    for i in range(map_start_indices[0] + 1):
      assert not loss_mask[i], f"Prefix token '{readable[i]}' at index {i} should have loss_mask=0"


def test_window_ts_values_correct():
  """TS values are correctly computed relative to window start."""
  builder, tokenizer = _make_builder()

  beatmap = Beatmap()
  # Object at 3000ms absolute time
  beatmap.hit_objects.append(Circle(x=0, y=0, time=3000))

  tokens, times, snaps = tokenizer.encode(beatmap)

  # Window starts at 1000ms
  window_tokens, loss_mask = builder.build(
    tokens=np.array(tokens),
    times=np.array(times, dtype=np.float64),
    snaps=np.array(snaps),
    audio_start_ms=1000,
    audio_end_ms=9000,
  )

  readable = [tokenizer.id_to_token[t] for t in window_tokens]

  # Object at 3000ms, window starts at 1000ms -> TS value = (3000-1000)/10 = 200
  ts_tokens = [t for t in readable if t.startswith("TS_")]
  assert len(ts_tokens) > 0, "No TS tokens found"

  # Find TS token before the OBJ_START
  for i, tok in enumerate(readable):
    if tok == "OBJ_START" and i >= 2:
      ts_tok = readable[i-2]
      assert ts_tok.startswith("TS_")
      ts_val = int(ts_tok.split("_")[1])
      expected_val = int(round((3000 - 1000) / 10))  # = 200
      assert ts_val == expected_val, \
        f"TS value {ts_val} != expected {expected_val} for object at 3000ms, window start 1000ms"


def test_window_overlap():
  """Overlapping windows share context tokens — context from early part of window."""
  builder, tokenizer = _make_builder(overlap=0.5)

  beatmap = Beatmap()
  # Objects spanning a range
  beatmap.hit_objects.append(Circle(x=0, y=0, time=1000))
  beatmap.hit_objects.append(Circle(x=100, y=100, time=3000))
  beatmap.hit_objects.append(Circle(x=200, y=200, time=5000))

  tokens, times, snaps = tokenizer.encode(beatmap)
  np_tokens = np.array(tokens)
  np_times = np.array(times, dtype=np.float64)
  np_snaps = np.array(snaps)

  # First window: 0-8000ms, predict_start=4000ms
  win1_tokens, win1_loss = builder.build(
    tokens=np_tokens, times=np_times, snaps=np_snaps,
    audio_start_ms=0, audio_end_ms=8000,
  )

  # Second window: 4000-12000ms, predict_start=8000ms
  win2_tokens, win2_loss = builder.build(
    tokens=np_tokens, times=np_times, snaps=np_snaps,
    audio_start_ms=4000, audio_end_ms=12000,
  )

  win1_readable = [tokenizer.id_to_token[t] for t in win1_tokens]
  win2_readable = [tokenizer.id_to_token[t] for t in win2_tokens]

  # Object at 5000ms should be in window 1's target (5000 >= 4000 and < 8000)
  # Object at 5000ms should be in window 2's context (5000 < 8000=predict_start)
  # Both windows should contain the object at 5000ms
  assert "OBJ_START" in win1_readable, "Window 1 should have objects"
  assert "OBJ_START" in win2_readable, "Window 2 should have objects"


def test_window_handles_empty():
  """Window with no objects in range produces a valid minimal sequence."""
  builder, tokenizer = _make_builder()

  beatmap = Beatmap()
  # Object far outside window range
  beatmap.hit_objects.append(Circle(x=0, y=0, time=100000))

  tokens, times, snaps = tokenizer.encode(beatmap)

  window_tokens, loss_mask = builder.build(
    tokens=np.array(tokens),
    times=np.array(times, dtype=np.float64),
    snaps=np.array(snaps),
    audio_start_ms=0,
    audio_end_ms=8000,
  )

  readable = [tokenizer.id_to_token[t] for t in window_tokens]

  # Should still have prefix tokens (SR, SPOS, MAP_START)
  assert any(t.startswith("SR_") for t in readable), "Should have SR token"
  assert "MAP_START" in readable, "Should have MAP_START"
  # No OBJ_START should be present (object is outside window range)
  # Note: it might still have MAP_END if the structural tokens are in range
  assert len(window_tokens) > 0, "Window should not be completely empty"


def test_window_max_tokens_respected():
  """Window output doesn't exceed max_tokens."""
  builder, tokenizer = _make_builder(max_tokens=64, overlap=0.5)

  beatmap = Beatmap()
  # Add many objects to exceed the token budget
  for i in range(50):
    beatmap.hit_objects.append(Circle(x=i*10 % 512, y=i*8 % 384, time=i * 100))

  tokens, times, snaps = tokenizer.encode(beatmap)

  window_tokens, loss_mask = builder.build(
    tokens=np.array(tokens),
    times=np.array(times, dtype=np.float64),
    snaps=np.array(snaps),
    audio_start_ms=0,
    audio_end_ms=8000,
  )

  assert len(window_tokens) <= 64, \
    f"Window has {len(window_tokens)} tokens, exceeding max_tokens=64"
  assert len(loss_mask) == len(window_tokens), \
    "Loss mask length must match window token length"


def test_window_token_loss_mask_alignment():
  """Window tokens and loss mask have the same length."""
  builder, tokenizer = _make_builder()

  beatmap = Beatmap()
  beatmap.timing_points.append(TimingPoint(time=0, beat_length=500, uninherited=1))
  beatmap.hit_objects.append(Circle(x=0, y=0, time=500))
  beatmap.hit_objects.append(Circle(x=100, y=100, time=1000))
  beatmap.hit_objects.append(Circle(x=200, y=200, time=1500))

  tokens, times, snaps = tokenizer.encode(beatmap)

  window_tokens, loss_mask = builder.build(
    tokens=np.array(tokens),
    times=np.array(times, dtype=np.float64),
    snaps=np.array(snaps),
    audio_start_ms=0,
    audio_end_ms=8000,
  )

  assert len(window_tokens) == len(loss_mask), \
    f"Token count ({len(window_tokens)}) != loss mask length ({len(loss_mask)})"
