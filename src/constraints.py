from .config import TokenizerConfig

def build_dsl_tokens(config: TokenizerConfig):
  DT_TOKENS = [ "DT_10", "DT_100", "DT_1000" ]
  X_TOKENS = [ f"X_{i}" for i in range(config.X_BINS) ]
  Y_TOKENS = [ f"Y_{i}" for i in range(config.Y_BINS) ]
  OBJ_TOKENS = [ "OBJ_START", "OBJ_END" ]
  TYPE_TOKENS = [ "T_CIRCLE", "T_SLIDER", "T_SPINNER" ]
  NC_TOKENS = [ "NC" ]
  SEG_TYPE_TOKENS = ["SEG_LINEAR", "SEG_BEZIER", "SEG_CAT", "SEG_PERFECT"]
  CP_TOKENS = [ f"CP_{i}" for i in range(config.SLIDER_CP_LIMIT) ]
  TP_TOKENS = [ "TP_START", "TP_END" ]
  SV_TOKENS = [ f"SV_{i}" for i in range(config.SLIDER_VEL_LIMIT * 10) ] # SV tokens are in 0.1 increments
  SR_TOKENS = [f"SR_{i}" for i in range(0, 11)]
  BPM_TOKENS = [f"BPM_{i}" for i in range(config.BPM_MIN, config.BPM_MAX + 1, config.BPM_JUMP)]
  STRUCTURAL_TOKENS = [ "MAP_START", "MAP_END", "EOS", "PAD" ]

  ALL_TOKENS: list[str] = (
    DT_TOKENS
    + X_TOKENS
    + Y_TOKENS
    + OBJ_TOKENS
    + TYPE_TOKENS
    + NC_TOKENS
    + SEG_TYPE_TOKENS
    + CP_TOKENS
    + TP_TOKENS
    + SV_TOKENS
    + SR_TOKENS
    + BPM_TOKENS
    + STRUCTURAL_TOKENS
  )

  return ALL_TOKENS


def build_vocab(config: TokenizerConfig) -> tuple[dict[str, int], list[str]]:
  tokens = build_dsl_tokens(config)

  token_to_id = { token: idx for idx, token in enumerate(tokens) }
  id_to_token = tokens

  return token_to_id, id_to_token