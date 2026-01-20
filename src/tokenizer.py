from .config import TokenizerConfig
from .constraints import build_vocab
from .osu import Beatmap, TimingPoint, Circle, Slider, Spinner, SliderCurve, CurveType, Difficulty

class Tokenizer:
  def __init__(self, config: TokenizerConfig):
    self.config = config
    self.token_to_id, self.id_to_token = build_vocab(config)
    self.x_bin_width = 512 / config.X_BINS
    self.y_bin_width = 384 / config.Y_BINS

  def encode(self, beatmap: Beatmap):
    tokens: list[int] = []
    beatmap_objects = sorted(
      beatmap.timing_points + beatmap.hit_objects,
      key=lambda obj: (obj.time, 0 if isinstance(obj, TimingPoint) else 1)
    )

    diff = beatmap.get_difficulty()
    sr_token = self.find_closest_token_from_vocab(diff.star_rating, "SR_")
    tokens.append(self.token_to_id[sr_token])

    tokens.append(self.token_to_id["MAP_START"])

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
        tokens.append(self.token_to_id["TP_START"])
        tokens.append(self.token_to_id[bpm_token])
        tokens.append(self.token_to_id["TP_END"])
      elif isinstance(obj, TimingPoint) and obj.uninherited == 0:
        svm = obj.get_slider_velocity_multiplier()
        sv_token = self.find_closest_token_from_vocab(svm * beatmap.difficulty.slider_multiplier, "SV_")
        tokens.append(self.token_to_id["TP_START"])
        tokens.append(self.token_to_id[sv_token])
        tokens.append(self.token_to_id["TP_END"])

      elif isinstance(obj, Circle):
        tokens.append(self.token_to_id["OBJ_START"])
        tokens.append(self.token_to_id["T_CIRCLE"])
        x_token = self.find_closest_token_from_vocab(obj.x / self.x_bin_width, "X_")
        y_token = self.find_closest_token_from_vocab(obj.y / self.y_bin_width, "Y_")
        tokens.append(self.token_to_id[x_token])
        tokens.append(self.token_to_id[y_token])
        tokens.append(self.token_to_id["OBJ_END"])

      elif isinstance(obj, Slider):
        tokens.append(self.token_to_id["OBJ_START"])
        tokens.append(self.token_to_id["T_SLIDER"])
        x_token = self.find_closest_token_from_vocab(obj.x / self.x_bin_width, "X_")
        y_token = self.find_closest_token_from_vocab(obj.y / self.y_bin_width, "Y_")
        tokens.append(self.token_to_id[x_token])
        tokens.append(self.token_to_id[y_token])
        len_token = self.find_closest_token_from_vocab(obj.object_params.length / self.config.SLIDER_LEN_BINS, "SL_")
        tokens.append(self.token_to_id[len_token])
        for curve in obj.object_params.curves:
          tokens.append(self.token_to_id[f"SEG_{curve.curve_type.name}"])
          for idx, (cp_x, cp_y) in enumerate(curve.curve_points):
            if idx >= self.config.SLIDER_CP_LIMIT:
              break
            cp_i_token = self.find_closest_token_from_vocab(idx, "CP_")
            tokens.append(self.token_to_id[cp_i_token])
            cp_x_token = self.find_closest_token_from_vocab(cp_x / self.x_bin_width, "X_")
            cp_y_token = self.find_closest_token_from_vocab(cp_y / self.y_bin_width, "Y_")
            tokens.append(self.token_to_id[cp_x_token])
            tokens.append(self.token_to_id[cp_y_token])
        tokens.append(self.token_to_id["OBJ_END"])

      elif isinstance(obj, Spinner):
        tokens.append(self.token_to_id["OBJ_START"])
        tokens.append(self.token_to_id["T_SPINNER"])
        spinning_duration = obj.object_params.end_time - obj.time + time_error
        tokens += self.encode_delta_time(spinning_duration)
        tokens.append(self.token_to_id["OBJ_END"])
    tokens.append(self.token_to_id["MAP_END"])
    tokens.append(self.token_to_id["EOS"])
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
    building_timing_point = None

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
          time += delta_ms
      elif token == "TP_START":
        building_timing_point = TimingPoint(time=time)
      elif token.startswith("SV_"):
        if building_timing_point:
          sv_multiplier = self.extract_number_from_token(token, "SV_")
          if sv_multiplier is None:
            continue
          building_timing_point.beat_length = -100.0 / (sv_multiplier / beatmap.difficulty.slider_multiplier)
          building_timing_point.uninherited = 0
      elif token.startswith("BPM_"):
        if building_timing_point:
          bpm = self.extract_number_from_token(token, "BPM_")
          if bpm is None:
            continue
          building_timing_point.beat_length = 60000.0 / bpm
          building_timing_point.uninherited = 1
      elif token == "TP_END":
        if building_timing_point:
          beatmap.timing_points.append(building_timing_point)
          building_timing_point = None
      elif token == "OBJ_START":
        continue
      elif token == "T_CIRCLE":
        building_object = Circle(time=time)
      elif token == "T_SLIDER":
        building_object = Slider(time=time)
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
      elif token.startswith("SL_") and building_slider_params:
        sl = self.extract_number_from_token(token, "SL_")
        if sl is not None:
          building_slider_params.length = sl * self.config.SLIDER_LEN_BINS
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
        building_object.object_params.end_time += delta_ms # type: ignore
      elif token == "OBJ_END" and building_object:
        beatmap.hit_objects.append(building_object)
        building_object = None
        building_slider_params = None
        building_slider_segment = None
        building_slider_control_point = None
        building_spinner_params = None
    return beatmap

  def extract_number_from_token(self, token: str, prefix: str) -> float | None:
    try:
      return float(token[len(prefix):])
    except:
      return None

  def find_closest_token_from_vocab(self, value: float, prefix: str) -> str:
    candidates = []

    for tok in self.token_to_id.keys():
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
    tokens = []

    while delta >= 1000:
      tokens.append(self.token_to_id["DT_1000"])
      delta -= 1000

    while delta >= 100:
      tokens.append(self.token_to_id["DT_100"])
      delta -= 100

    while delta >= 10:
      tokens.append(self.token_to_id["DT_10"])
      delta -= 10

    return tokens