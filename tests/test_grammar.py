from src.grammar import Grammar, GrammarState, Tok
from src.tokenizer import Tokenizer
from .test_tokenizer import test_tokens
from src.config import ExperimentConfig

def test_grammar_allowed_next_tokens():
  config = ExperimentConfig()
  tokenizer = Tokenizer(config.tokenizer)
  grammar = Grammar(tokenizer)

  state = GrammarState()

  for i, token in enumerate(test_tokens):
    token_id = tokenizer.vocab[token]
    allowed_tokens = grammar.allowed_next_tokens(state)

    assert token_id in allowed_tokens, \
      f"Token '{token}' (id={token_id}) not allowed in current state.\n" \
      f"Context: {test_tokens[max(0, i-5):i+1]}\n" \
      f"Allowed prefixes: {set(tokenizer.id_to_token[t].split('_')[0] for t in allowed_tokens)}"

    state = grammar.update_state(state, token_id)


def test_grammar_monotonic_ts():
  """TS tokens must be monotonically non-decreasing."""
  config = ExperimentConfig()
  tokenizer = Tokenizer(config.tokenizer)
  grammar = Grammar(tokenizer)

  state = GrammarState()
  state.in_map = True

  # Emit TS_100 SNAP_0
  ts_100 = tokenizer.vocab["TS_100"]
  snap_0 = tokenizer.vocab["SNAP_0"]
  obj_start = tokenizer.vocab["OBJ_START"]
  obj_end = tokenizer.vocab["OBJ_END"]
  t_circle = tokenizer.vocab["T_CIRCLE"]
  x_0 = tokenizer.vocab["X_0"]
  y_0 = tokenizer.vocab["Y_0"]

  state = grammar.update_state(state, ts_100)
  state = grammar.update_state(state, snap_0)

  # Place an object
  state = grammar.update_state(state, obj_start)
  state = grammar.update_state(state, t_circle)
  state = grammar.update_state(state, x_0)
  state = grammar.update_state(state, y_0)
  state = grammar.update_state(state, obj_end)

  # Now allowed TS tokens should only be >= 100
  allowed = grammar.allowed_next_tokens(state)
  ts_50 = tokenizer.vocab["TS_50"]
  ts_100_check = tokenizer.vocab["TS_100"]
  ts_200 = tokenizer.vocab["TS_200"]

  assert ts_50 not in allowed, "TS_50 should not be allowed after TS_100 (monotonic)"
  assert ts_100_check in allowed, "TS_100 should be allowed (equal)"
  assert ts_200 in allowed, "TS_200 should be allowed (greater)"


def test_grammar_ts_snap_sequence():
  """After TS, SNAP must immediately follow."""
  config = ExperimentConfig()
  tokenizer = Tokenizer(config.tokenizer)
  grammar = Grammar(tokenizer)

  state = GrammarState()
  state.in_map = True

  ts_50 = tokenizer.vocab["TS_50"]
  state = grammar.update_state(state, ts_50)

  # After TS, only SNAP tokens should be allowed
  allowed = grammar.allowed_next_tokens(state)
  for tid in allowed:
    tok_str = tokenizer.id_to_token[tid]
    assert tok_str.startswith("SNAP_"), \
      f"After TS token, only SNAP should be allowed, got '{tok_str}'"


def test_grammar_after_snap():
  """After SNAP, BPM or OBJ_START should be allowed."""
  config = ExperimentConfig()
  tokenizer = Tokenizer(config.tokenizer)
  grammar = Grammar(tokenizer)

  state = GrammarState()
  state.in_map = True

  # TS -> SNAP
  state = grammar.update_state(state, tokenizer.vocab["TS_50"])
  state = grammar.update_state(state, tokenizer.vocab["SNAP_4"])

  # Now should allow BPM, OBJ_START, TS, MAP_END (back to in-map, not-in-object)
  allowed = grammar.allowed_next_tokens(state)
  allowed_strs = {tokenizer.id_to_token[t] for t in allowed}

  assert "OBJ_START" in allowed_strs
  assert any(s.startswith("BPM_") for s in allowed_strs)


def test_grammar_spos_before_map():
  """SPOS tokens should be allowed in the pre-map state."""
  config = ExperimentConfig()
  tokenizer = Tokenizer(config.tokenizer)
  grammar = Grammar(tokenizer)

  state = GrammarState()
  allowed = grammar.allowed_next_tokens(state)
  allowed_strs = {tokenizer.id_to_token[t] for t in allowed}

  assert any(s.startswith("SPOS_") for s in allowed_strs)
  assert any(s.startswith("SR_") for s in allowed_strs)
  assert "MAP_START" in allowed_strs


def test_grammar_sustain_in_spinner():
  """SUSTAIN should be allowed inside spinner objects."""
  config = ExperimentConfig()
  tokenizer = Tokenizer(config.tokenizer)
  grammar = Grammar(tokenizer)

  state = GrammarState()
  state.in_map = True
  state.in_object = True
  state.object_type = Tok.T_SPINNER

  allowed = grammar.allowed_next_tokens(state)
  allowed_strs = {tokenizer.id_to_token[t] for t in allowed}

  assert "SUSTAIN" in allowed_strs
  assert "OBJ_END" in allowed_strs
  assert any(s.startswith("DT_") for s in allowed_strs)


def test_grammar_dt_not_at_top_level():
  """DT tokens should NOT be allowed at the top level (between objects)."""
  config = ExperimentConfig()
  tokenizer = Tokenizer(config.tokenizer)
  grammar = Grammar(tokenizer)

  state = GrammarState()
  state.in_map = True

  allowed = grammar.allowed_next_tokens(state)
  allowed_strs = {tokenizer.id_to_token[t] for t in allowed}

  dt_at_top = [s for s in allowed_strs if s.startswith("DT_")]
  assert len(dt_at_top) == 0, f"DT tokens should not be at top level, found: {dt_at_top}"
