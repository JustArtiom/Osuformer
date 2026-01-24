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

    print()
    print(f"{test_tokens[max(0, i-10):i]}")
    print(f"Allowed: {set(["".join(tokenizer.id_to_token[t].split('_')[0]) for t in allowed_tokens])}")

    assert tokenizer.id_to_token[token_id] in [tokenizer.id_to_token[t] for t in allowed_tokens], \
      f"Token '{token}' not allowed in current state.\nContext {test_tokens[max(0, i-5):i+1]}"

    state = grammar.update_state(state, token_id)