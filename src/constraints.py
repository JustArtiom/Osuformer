from .config import TokenizerConfig
from .osu import MapStyle
from dataclasses import dataclass

@dataclass
class DSLTokens:
  PAD_TOKENS: list[str]
  STYLE_TOKENS: list[str]
  DT_TOKENS: list[str]
  X_TOKENS: list[str]
  Y_TOKENS: list[str]
  OBJ_TOKENS: list[str]
  TYPE_TOKENS: list[str]
  SEG_TYPE_TOKENS: list[str]
  SL_TOKENS: list[str]
  CP_TOKENS: list[str]
  TP_TOKENS: list[str]
  SV_TOKENS: list[str]
  SR_TOKENS: list[str]
  BPM_TOKENS: list[str]
  STRUCTURAL_TOKENS: list[str]
  
def build_dsl_tokens(config: TokenizerConfig):
  dsl_tokens = DSLTokens(
    PAD_TOKENS = ["PAD"],
    STYLE_TOKENS = [f"STYLE_{style.name}" for style in MapStyle],
    DT_TOKENS = [ "DT_10", "DT_100", "DT_1000" ],
    X_TOKENS = [ f"X_{i}" for i in range(config.X_BINS) ],
    Y_TOKENS = [ f"Y_{i}" for i in range(config.Y_BINS) ],
    OBJ_TOKENS = [ "OBJ_START", "OBJ_END" ],
    TYPE_TOKENS = [ "T_CIRCLE", "T_SLIDER", "T_SPINNER" ],
    SEG_TYPE_TOKENS = ["SEG_LINEAR", "SEG_BEZIER", "SEG_CAT", "SEG_PERFECT"],
    SL_TOKENS = [ f"SL_{i}" for i in range(0, config.SLIDER_LEN_MAX, config.SLIDER_LEN_BINS) ],
    CP_TOKENS = [ f"CP_{i}" for i in range(config.SLIDER_CP_LIMIT) ],
    TP_TOKENS = [ "TP_START", "TP_END" ],
    SV_TOKENS = [ f"SV_{i/10:.1f}" for i in range(config.SLIDER_VEL_LIMIT * 10) ], # SV tokens are in 0.1 increments,
    SR_TOKENS = [f"SR_{i}" for i in range(0, 11)],
    BPM_TOKENS = [f"BPM_{i}" for i in range(config.BPM_MIN, config.BPM_MAX + 1, config.BPM_JUMP)],
    STRUCTURAL_TOKENS = [ "MAP_START", "MAP_END", "EOS", "BOS" ]
  )

  ALL_TOKENS: list[str] = (
    dsl_tokens.PAD_TOKENS
    + dsl_tokens.STYLE_TOKENS
    + dsl_tokens.DT_TOKENS
    + dsl_tokens.X_TOKENS
    + dsl_tokens.Y_TOKENS
    + dsl_tokens.OBJ_TOKENS
    + dsl_tokens.TYPE_TOKENS
    + dsl_tokens.SEG_TYPE_TOKENS
    + dsl_tokens.SL_TOKENS
    + dsl_tokens.CP_TOKENS
    + dsl_tokens.TP_TOKENS
    + dsl_tokens.SV_TOKENS
    + dsl_tokens.SR_TOKENS
    + dsl_tokens.BPM_TOKENS
    + dsl_tokens.STRUCTURAL_TOKENS
  )

  return ALL_TOKENS, dsl_tokens


def build_vocab(config: TokenizerConfig) -> tuple[dict[str, int], list[str], DSLTokens]:
  tokens, dsl_tokens = build_dsl_tokens(config)

  token_to_id = { token: idx for idx, token in enumerate(tokens) }
  id_to_token = tokens

  return token_to_id, id_to_token, dsl_tokens