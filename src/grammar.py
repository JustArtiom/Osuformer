from dataclasses import dataclass
from typing import Optional
from .tokenizer import Tokenizer
from .constraints import Tok

@dataclass
class GrammarState:
    in_map: bool = False
    in_object: bool = False

    object_type: Optional[str] = None

    # TS/SNAP sequencing
    expect_snap: bool = False       # after TS, must emit SNAP
    last_ts_value: int = -1         # for monotonic time constraint

    # generic expectations
    expect_x: bool = False
    expect_y: bool = False

    # slider-specific state
    slider_has_base_xy: bool = False
    slider_has_sv: bool = False
    slider_has_length: bool = False
    slider_has_slides: bool = False
    slider_has_segment: bool = False
    slider_expect_cp_xy: Optional[str] = None  # None | "X" | "Y"

class Grammar():
  def __init__(self, tokenizer: Tokenizer):
    self.tok = tokenizer.vocab
    self.rev = tokenizer.id_to_token
    self.groups = {
      "SR":       self._by_prefix("SR_"),
      "STYLE":    self._by_prefix("STYLE_"),
      "TS":       self._by_prefix("TS_"),
      "SNAP":     self._by_prefix("SNAP_"),
      "SPOS":     self._by_prefix("SPOS_"),
      "DT":       self._by_prefix("DT_"),
      "X":        self._by_prefix("X_"),
      "Y":        self._by_prefix("Y_"),
      "BPM":      self._by_prefix("BPM_"),
      "SV":       self._by_prefix("SV_"),
      "CP":       self._by_prefix("CP_"),
      "SL":       self._by_prefix("SL_"),
      "SLIDES":   self._by_prefix("SLIDES_"),
      "SEG":      self._by_prefix("SEG_"),
    }
    # SUSTAIN is a single token, handle specially
    self._sustain_id = self.tok.get(Tok.SUSTAIN, -1)

    # Precompute TS value by token id for monotonic filtering
    self._ts_value_by_id: dict[int, int] = {}
    for t, tid in self.tok.items():
      if t.startswith(Tok.TS):
        try:
          self._ts_value_by_id[tid] = int(float(t[len(Tok.TS):]))
        except ValueError:
          pass

  def _by_prefix(self, prefix):
    return {
      idx for tok, idx in self.tok.items()
      if tok.startswith(prefix)
    }

  def initial_state(self) -> GrammarState:
    return GrammarState()

  def update_state(self, state: GrammarState, token_id: int) -> GrammarState:
    tok = self.rev[token_id]

    if tok == Tok.MAP_START:
      state.in_map = True
      state.last_ts_value = -1

    elif tok == Tok.MAP_END:
      state.in_map = False

    elif tok.startswith(Tok.TS):
      val = self._ts_value_by_id.get(token_id, 0)
      state.last_ts_value = val
      state.expect_snap = True

    elif tok.startswith(Tok.SNAP):
      state.expect_snap = False

    elif tok.startswith(Tok.SPOS):
      pass  # no state change

    elif tok == Tok.SUSTAIN:
      pass  # no state change

    # OBJECT
    elif tok == Tok.OBJ_START:
      state.in_object = True
      state.object_type = None
      state.slider_has_base_xy = False
      state.slider_has_sv = False
      state.slider_has_length = False
      state.slider_has_slides = False
      state.slider_has_segment = False
      state.slider_expect_cp_xy = None

    elif tok == Tok.OBJ_END:
      state.in_object = False
      state.object_type = None
      state.expect_x = False
      state.expect_y = False
      state.slider_has_base_xy = False
      state.slider_has_sv = False
      state.slider_has_length = False
      state.slider_has_slides = False
      state.slider_has_segment = False
      state.slider_expect_cp_xy = None

    elif tok in (Tok.T_CIRCLE, Tok.T_SLIDER, Tok.T_SPINNER):
      state.object_type = tok

      if tok == Tok.T_SLIDER:
        state.expect_x = True
        state.expect_y = False
        state.slider_has_base_xy = False
        state.slider_has_sv = False
        state.slider_has_length = False
        state.slider_has_slides = False
        state.slider_has_segment = False
        state.slider_expect_cp_xy = None

      elif tok == Tok.T_CIRCLE:
        state.expect_x = True
        state.expect_y = False

      elif tok == Tok.T_SPINNER:
        state.expect_x = False
        state.expect_y = False

    elif tok.startswith("X_"):
      if state.object_type == Tok.T_SLIDER and not state.slider_has_base_xy and not state.slider_expect_cp_xy:
        state.expect_x = False
        state.expect_y = True
      elif state.slider_expect_cp_xy == "X":
        state.slider_expect_cp_xy = "Y"
      else:
        state.expect_x = False
        state.expect_y = True

    elif tok.startswith("Y_"):
      if state.object_type == Tok.T_SLIDER and not state.slider_has_base_xy and not state.slider_expect_cp_xy:
        state.expect_y = False
        state.slider_has_base_xy = True
      elif state.slider_expect_cp_xy == "Y":
        state.slider_expect_cp_xy = None
      else:
        state.expect_y = False

    elif tok.startswith("SV_"):
      if state.object_type == Tok.T_SLIDER and state.slider_has_base_xy and not state.slider_expect_cp_xy:
        state.slider_has_sv = True

    elif tok.startswith("SL_"):
      state.slider_has_length = True
    elif tok.startswith("SLIDES_"):
      state.slider_has_slides = True

    elif tok.startswith("SEG_"):
      state.slider_has_segment = True

    elif tok.startswith("CP_"):
      state.slider_expect_cp_xy = "X"

    return state

  def allowed_next_tokens(self, state: GrammarState) -> set[int]:
    # Pre-map: SR, STYLE, SPOS, MAP_START, EOS
    if not state.in_map:
      return (
        self.groups["SR"]
        | self.groups["STYLE"]
        | self.groups["SPOS"]
        | {self.tok[Tok.MAP_START]}
        | {self.tok[Tok.EOS]}
      )

    # After TS, must emit SNAP
    if state.expect_snap:
      return self.groups["SNAP"]

    # Inside an object
    if state.in_object:
      if state.object_type is None:
        return {
          self.tok[Tok.T_CIRCLE],
          self.tok[Tok.T_SLIDER],
          self.tok[Tok.T_SPINNER],
        }

      if state.expect_x:
        return self.groups["X"]

      if state.expect_y:
        return self.groups["Y"]

      if state.object_type == Tok.T_CIRCLE:
        return {self.tok[Tok.OBJ_END]}

      if state.object_type == Tok.T_SPINNER:
        allowed = set(self.groups["DT"])
        allowed.add(self.tok[Tok.OBJ_END])
        if self._sustain_id >= 0:
          allowed.add(self._sustain_id)
        return allowed

      if state.object_type == Tok.T_SLIDER:
        if not state.slider_has_sv and state.slider_has_base_xy:
          return self.groups["SV"]
        if not state.slider_has_length:
          return self.groups["SL"]
        if not state.slider_has_slides:
          return self.groups["SLIDES"]

        if not state.slider_has_segment:
          return self.groups["SEG"]

        if state.slider_expect_cp_xy == "X":
          return self.groups["X"]
        if state.slider_expect_cp_xy == "Y":
          return self.groups["Y"]

        allowed = set(self.groups["SEG"])

        if state.slider_has_segment and state.slider_expect_cp_xy is None:
            allowed |= self.groups["CP"]
            allowed.add(self.tok[Tok.OBJ_END])
            if self._sustain_id >= 0:
              allowed.add(self._sustain_id)

        return allowed

    # In map, not in object: TS (with monotonic constraint) or MAP_END
    allowed_ts = self._monotonic_ts_tokens(state.last_ts_value)
    return (
        allowed_ts
        | self.groups["BPM"]
        | {self.tok[Tok.OBJ_START]}
        | {self.tok[Tok.MAP_END]}
    )

  def _monotonic_ts_tokens(self, last_ts_value: int) -> set[int]:
    """Return TS token IDs with values >= last_ts_value (monotonic constraint)."""
    if last_ts_value < 0:
      return self.groups["TS"]
    return {
      tid for tid, val in self._ts_value_by_id.items()
      if val >= last_ts_value
    }

  def consume(self, state: GrammarState, token_id: int):
    allowed = self.allowed_next_tokens(state)

    if token_id not in allowed:
        token = self.rev[token_id]
        allowed_tokens = [self.rev[i] for i in sorted(allowed)]
        raise AssertionError(
            f"Token '{token}' not allowed in current state.\n"
            f"Allowed: {allowed_tokens}"
        )

    return self.update_state(state, token_id)
