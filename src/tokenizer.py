from typing import Optional, List
from .config import TokenizerConfig
from .constraints import build_vocab
from .osu import Beatmap, TimingPoint, Circle, Slider, Spinner, SliderCurve, CurveType, Difficulty

class Tokenizer:
  def __init__(self, config: Optional[TokenizerConfig] = None, with_styles: bool = True):
    if config is None:
      return
    self.config = config
    self.token_to_id, self.id_to_token, self.DSL_TOKENS = build_vocab(config)
    self.vocab = self.token_to_id
    self.x_bin_width = 512 / config.X_BINS
    self.y_bin_width = 384 / config.Y_BINS
    self.with_styles = with_styles

    dt_bins: list[tuple[int, int]] = []
    for tok, tok_id in self.vocab.items():
        if tok.startswith("DT_"):
            size = self.extract_number_from_token(tok, "DT_")
            assert size is not None
            size *= self.config.DT_BIN_MS
            if size is not None and size > 0:
                dt_bins.append((int(size), tok_id))
    dt_bins.sort(reverse=True, key=lambda x: x[0])
    self.dt_bins = dt_bins

  def encode(self, beatmap):
    tokens: List[int] = []
    times: List[float] = []

    current_time_ms = 0

    for token, idx in [(self.id_to_token[i], i) for i in self.encode_tokens(beatmap)]:
        tokens.append(idx)
        times.append(current_time_ms)

        if token.startswith("DT_"):
            delta_ms = self.extract_number_from_token(token, "DT_")
            if delta_ms is not None:
                current_time_ms += delta_ms * self.config.DT_BIN_MS

    return tokens, times

  def encode_tokens(self, beatmap: Beatmap):
    tokens: list[int] = []
    beatmap_objects = sorted(
      [tp for tp in beatmap.timing_points if tp.uninherited == 1] + beatmap.hit_objects,
      key=lambda obj: (obj.time, 0 if isinstance(obj, TimingPoint) else 1)
    )

    diff = beatmap.get_difficulty()
    sr_token = self.find_closest_token_from_vocab(diff.star_rating, "SR_")
    tokens.append(self.vocab[sr_token])
    styles = beatmap.styles
    if self.with_styles:
      for style in styles:
        style_token = f"STYLE_{style.name}"
        if style_token in self.vocab:
          tokens.append(self.vocab[style_token])

    tokens.append(self.vocab["MAP_START"])

    last_time = 0
    time_error = 0.0

    for obj in beatmap_objects:
      raw_delta = obj.time - last_time
      raw_delta += time_error
      quantized = int(round(raw_delta / self.config.DT_BIN_MS)) * self.config.DT_BIN_MS
      time_error = raw_delta - quantized
      last_time = obj.time
      tokens += self.encode_delta_time(quantized)

      if isinstance(obj, TimingPoint) and obj.uninherited == 1:
        bpm = obj.get_bpm()
        bpm_token = self.find_closest_token_from_vocab(bpm, "BPM_")
        tokens.append(self.vocab[bpm_token])

      elif isinstance(obj, Circle):
        tokens.append(self.vocab["OBJ_START"])
        tokens.append(self.vocab["T_CIRCLE"])
        x_token = self.find_closest_token_from_vocab(obj.x / self.x_bin_width, "X_")
        y_token = self.find_closest_token_from_vocab(obj.y / self.y_bin_width, "Y_")
        tokens.append(self.vocab[x_token])
        tokens.append(self.vocab[y_token])
        tokens.append(self.vocab["OBJ_END"])

      elif isinstance(obj, Slider):
        tokens.append(self.vocab["OBJ_START"])
        tokens.append(self.vocab["T_SLIDER"])
        x_token = self.find_closest_token_from_vocab(obj.x / self.x_bin_width, "X_")
        y_token = self.find_closest_token_from_vocab(obj.y / self.y_bin_width, "Y_")
        tokens.append(self.vocab[x_token])
        tokens.append(self.vocab[y_token])
        sv_multiplier = beatmap.get_slider_velocity_multiplier_at(obj.time) * beatmap.difficulty.slider_multiplier
        sv_token = self.find_closest_token_from_vocab(sv_multiplier, "SV_")
        tokens.append(self.vocab[sv_token])
        len_token = self.find_closest_token_from_vocab(obj.object_params.length, "SL_")
        tokens.append(self.vocab[len_token])
        slides_token = self.find_closest_token_from_vocab(obj.object_params.slides, "SLIDES_")
        tokens.append(self.vocab[slides_token])
        for curve in obj.object_params.curves:
          tokens.append(self.vocab[f"SEG_{curve.curve_type.name}"])
          for idx, (cp_x, cp_y) in enumerate(curve.curve_points):
            if idx >= self.config.SLIDER_CP_LIMIT:
              break
            cp_i_token = self.find_closest_token_from_vocab(idx, "CP_")
            tokens.append(self.vocab[cp_i_token])
            cp_x_token = self.find_closest_token_from_vocab(cp_x / self.x_bin_width, "X_")
            cp_y_token = self.find_closest_token_from_vocab(cp_y / self.y_bin_width, "Y_")
            tokens.append(self.vocab[cp_x_token])
            tokens.append(self.vocab[cp_y_token])
        tokens.append(self.vocab["OBJ_END"])

      elif isinstance(obj, Spinner):
        tokens.append(self.vocab["OBJ_START"])
        tokens.append(self.vocab["T_SPINNER"])
        spinning_duration = (obj.object_params.end_time - obj.time) + time_error
        quantized = int(round(spinning_duration / self.config.DT_BIN_MS)) * self.config.DT_BIN_MS
        time_error = spinning_duration - quantized
        tokens += self.encode_delta_time(quantized)
        tokens.append(self.vocab["OBJ_END"])
        last_time = obj.object_params.end_time
    tokens.append(self.vocab["MAP_END"])
    tokens.append(self.vocab["EOS"])
    return tokens

  def decode(self, tokens: list[int]) -> Beatmap:
    readable_tokens = [self.id_to_token[t] for t in tokens]
    beatmap = Beatmap(
      difficulty=Difficulty(
        slider_multiplier=1.0
      )
    )

    time = 0
    building_object = None
    building_slider_params = None
    building_slider_segment = None
    building_slider_control_point = None
    building_spinner_params = None
    building_slider_sv = None

    for i, token in enumerate(readable_tokens):
      if token == "MAP_START":
        continue
      elif token == "MAP_END":
        continue
      elif token == "EOS":
        continue
      elif token.startswith("SR_"):
        continue
      elif token.startswith("DT_") and not building_object:
        delta_ms = self.extract_number_from_token(token, "DT_")
        if delta_ms is not None:
          time += delta_ms * self.config.DT_BIN_MS
      elif token.startswith("BPM_"):
        if not building_object:
          bpm = self.extract_number_from_token(token, "BPM_")
          if bpm is None:
            continue
          timing_point = TimingPoint(time=time)
          timing_point.beat_length = 60000.0 / bpm
          timing_point.uninherited = 1
          beatmap.timing_points.append(timing_point)
      elif token == "OBJ_START":
        continue
      elif token == "T_CIRCLE":
        building_object = Circle(time=time)
      elif token == "T_SLIDER":
        building_object = Slider(time=time)
        building_slider_sv = None
      elif token == "T_SPINNER":
        building_object = Spinner(time=time)
        building_spinner_params = building_object.object_params
      elif token.startswith("X_") and building_object and not building_slider_params and not building_spinner_params:
        x = self.extract_number_from_token(token, "X_")
        if x is not None:
          x *= self.x_bin_width
          building_object.x = x
      elif token.startswith("Y_") and building_object and not building_slider_params and not building_spinner_params:
        y = self.extract_number_from_token(token, "Y_")
        if y is not None:
          y *= self.y_bin_width
          building_object.y = y
        if isinstance(building_object, Slider):
          building_slider_params = building_object.object_params
      elif token.startswith("SV_") and building_slider_params:
        sv_multiplier = self.extract_number_from_token(token, "SV_")
        if sv_multiplier is not None:
          building_slider_sv = sv_multiplier
      elif token.startswith("SL_") and building_slider_params:
        sl = self.extract_number_from_token(token, "SL_")
        if sl is not None:
          building_slider_params.length = sl
      elif token.startswith("SLIDES_") and building_slider_params:
        slides = self.extract_number_from_token(token, "SLIDES_")
        if slides is not None:
          building_slider_params.slides = int(slides)
      elif token.startswith("SEG_") and building_slider_params:
        curve_type_str = token[len("SEG_"):]
        curve_type = CurveType[curve_type_str]
        building_slider_segment = SliderCurve(curve_type=curve_type, curve_points=[])
        building_slider_params.curves.append(building_slider_segment)
      elif token.startswith("CP_") and building_slider_segment:
        building_slider_control_point = ()
      elif token.startswith("X_") and building_slider_control_point is not None:
        cp_x = self.extract_number_from_token(token, "X_")
        if cp_x is None:
          continue
        cp_x *= self.x_bin_width
        building_slider_control_point += (cp_x,)
      elif token.startswith("Y_") and building_slider_control_point is not None:
        cp_y = self.extract_number_from_token(token, "Y_")
        if cp_y is None:
          continue
        cp_y *= self.y_bin_width
        building_slider_control_point += (cp_y,)
        if len(building_slider_control_point) == 2:
          building_slider_segment.curve_points.append(building_slider_control_point) # type: ignore
          building_slider_control_point = None
          if(not readable_tokens[i+1].startswith("CP_")):
            building_slider_segment = None
            building_slider_params = None
      elif token.startswith("DT_") and building_spinner_params:
        delta_ms = self.extract_number_from_token(token, "DT_")
        if delta_ms is None:
          continue
        delta_ms *= self.config.DT_BIN_MS
        building_object.object_params.end_time += delta_ms # type: ignore
        time += delta_ms
      elif token == "OBJ_END" and building_object:
        if isinstance(building_object, Slider):
          sv_multiplier = building_slider_sv if building_slider_sv is not None else 1.0
          timing_point = TimingPoint(time=building_object.time)
          timing_point.beat_length = -100.0 / (sv_multiplier / beatmap.difficulty.slider_multiplier)
          timing_point.uninherited = 0
          beatmap.timing_points.append(timing_point)
        beatmap.hit_objects.append(building_object)
        building_object = None
        building_slider_params = None
        building_slider_segment = None
        building_slider_control_point = None
        building_spinner_params = None
        building_slider_sv = None

    beatmap._recalculate_slider_durations()  
    return beatmap

  def extract_number_from_token(self, token: str, prefix: str) -> float | None:
    try:
      return float(token[len(prefix):])
    except:
      return None

  def find_closest_token_from_vocab(self, value: float, prefix: str) -> str:
    candidates = []

    for tok in self.vocab.keys():
        if tok.startswith(prefix):
            num = self.extract_number_from_token(tok, prefix)
            if num is not None:
              candidates.append((tok, num))

    if not candidates:
        raise ValueError(f"No tokens found with prefix {prefix}")

    closest_tok, _ = min(candidates, key=lambda x: abs(x[1] - value))
    return closest_tok

  def encode_delta_time(self, delta_ms: float) -> list[int]:
    delta = int(round(delta_ms / self.config.DT_BIN_MS)) * self.config.DT_BIN_MS
    tokens: list[int] = []

    for size, tok_id in self.dt_bins:
        if delta <= 0:
            break
        count, delta = divmod(delta, size)
        if count:
            tokens.extend([tok_id] * count)

    return tokens
  
  def save(self, filepath: str):
    import json
    with open(filepath, "w") as f:
      json.dump(self.vocab, f, indent=2)

  def load(self, filepath: str):
    import json
    with open(filepath, "r") as f:
      vocab = json.load(f)
    self.vocab = vocab
    self.token_to_id = vocab
    self.id_to_token = ["" for _ in range(len(vocab))]
    for token, idx in vocab.items():
      self.id_to_token[idx] = token
    return self
