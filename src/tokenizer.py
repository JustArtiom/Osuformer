from .config import TokenizerConfig
from .constraints import build_vocab
from .osu import Beatmap, TimingPoint, Circle, Slider, Spinner, CurveType

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
    lastTime = 0

    for obj in beatmap_objects:
      delta_ms = obj.time - lastTime
      lastTime = obj.time
      tokens += self.encode_delta_time(delta_ms)

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
        spinning_duration = obj.object_params.end_time - obj.time
        tokens += self.encode_delta_time(spinning_duration)
        tokens.append(self.token_to_id["OBJ_END"])
    tokens.append(self.token_to_id["MAP_END"])
    tokens.append(self.token_to_id["EOS"])
    return tokens

  def decode(self, tokens: list[int]) -> Beatmap:
    raise NotImplementedError("Decoding is not implemented yet.")

  def find_closest_token_from_vocab(self, value: float, prefix: str) -> str:
    candidates = []

    for tok in self.token_to_id.keys():
        if tok.startswith(prefix):
            try:
                num = float(tok[len(prefix):])
                candidates.append((tok, num))
            except:
                continue

    if not candidates:
        raise ValueError(f"No tokens found with prefix {prefix}")

    closest_tok, _ = min(candidates, key=lambda x: abs(x[1] - value))
    return closest_tok

  def encode_delta_time(self, delta_ms: float) -> list[str]:
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

    if len(tokens) < 1:
      tokens.append(self.token_to_id["DT_0"])

    return tokens