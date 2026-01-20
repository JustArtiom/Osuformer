from src.osu import (
  Beatmap,
  TimingPoint,
  Difficulty,
  Circle,
  Slider,
  Spinner,
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