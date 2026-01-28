from .config import TokenizerConfig
from .osu import MapStyle
from dataclasses import dataclass
from typing import Union

class Tok:
  STYLE   = "STYLE_"
  SR      = "SR_"
  DT      = "DT_"
  X       = "X_"
  Y       = "Y_"
  BPM     = "BPM_"
  SV      = "SV_"
  CP      = "CP_"
  SL      = "SL_"
  SEG     = "SEG_"

  MAP_START = "MAP_START"
  MAP_END   = "MAP_END"
  EOS       = "EOS"
  BOS       = "BOS"
  
  TP_START  = "TP_START"
  TP_END    = "TP_END"

  OBJ_START = "OBJ_START"
  OBJ_END   = "OBJ_END"

  T_CIRCLE  = "T_CIRCLE"
  T_SLIDER  = "T_SLIDER"
  T_SPINNER = "T_SPINNER"


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
    STYLE_TOKENS = [tok(Tok.STYLE, style.name) for style in MapStyle],
    DT_TOKENS = [tok(Tok.DT, i) for i in range(1, int(config.DT_MAX_MS / config.DT_BIN_MS) + 1)],
    X_TOKENS = [tok(Tok.X, i) for i in range(config.X_BINS)],
    Y_TOKENS = [tok(Tok.Y, i) for i in range(config.Y_BINS)],
    OBJ_TOKENS = [tok(Tok.OBJ_START), tok(Tok.OBJ_END)],
    TYPE_TOKENS = [tok(Tok.T_CIRCLE), tok(Tok.T_SLIDER), tok(Tok.T_SPINNER)],
    SEG_TYPE_TOKENS = [tok(Tok.SEG, seg_type) for seg_type in ["LINEAR", "BEZIER", "CAT", "PERFECT"]],
    SL_TOKENS = [tok(Tok.SL, i) for i in range(0, config.SLIDER_LEN_MAX, config.SLIDER_LEN_BINS)],
    CP_TOKENS = [tok(Tok.CP, i) for i in range(config.SLIDER_CP_LIMIT)],
    TP_TOKENS = [tok(Tok.TP_START), tok(Tok.TP_END)],
    SV_TOKENS = [tok(Tok.SV, i/10) for i in range(1, config.SLIDER_VEL_LIMIT * 10 + 1)], # SV tokens are in 0.1 increments,
    SR_TOKENS = [tok(Tok.SR, i) for i in range(0, 11)],
    BPM_TOKENS = [tok(Tok.BPM, i) for i in range(config.BPM_MIN, config.BPM_MAX + 1, config.BPM_JUMP)],
    STRUCTURAL_TOKENS = [tok(Tok.MAP_START), tok(Tok.MAP_END), tok(Tok.EOS), tok(Tok.BOS)]
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

def tok(tok: str, value: Union[str, int, float] = "") -> str:
  return f"{tok}{value}"

def build_vocab(config: TokenizerConfig) -> tuple[dict[str, int], list[str], DSLTokens]:
  tokens, dsl_tokens = build_dsl_tokens(config)

  token_to_id = { token: idx for idx, token in enumerate(tokens) }
  id_to_token = tokens

  return token_to_id, id_to_token, dsl_tokens
