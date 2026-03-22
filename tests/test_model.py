import torch
from src.config import ExperimentConfig
from src.model.model import build_model
from src.tokenizer import Tokenizer


def _make_small_model():
  """Build a small model for fast testing."""
  config = ExperimentConfig()
  # Use tiny dimensions for speed
  config.model.encoder.d_model = 32
  config.model.encoder.layers = 1
  config.model.encoder.heads = 2
  config.model.encoder.ffn_dim = 64
  config.model.encoder.conv_kernel = 3
  config.model.encoder.dropout = 0.0
  config.model.decoder.d_model = 32
  config.model.decoder.layers = 1
  config.model.decoder.heads = 2
  config.model.decoder.ffn_dim = 64
  config.model.decoder.dropout = 0.0
  config.audio.n_mels = 16  # smaller mel dimension for testing

  tokenizer = Tokenizer(config.tokenizer)
  vocab_size = len(tokenizer.vocab)

  model = build_model(config, vocab_size)
  model.eval()
  return model, tokenizer, config


def test_model_forward():
  """Forward pass with conditioning produces correct output shape."""
  model, tokenizer, config = _make_small_model()
  vocab_size = len(tokenizer.vocab)

  B, T_mel, T_tgt = 2, 100, 20

  src = torch.randn(B, T_mel, config.audio.n_mels)
  tgt = torch.randint(0, vocab_size, (B, T_tgt))
  conditioning = torch.tensor([[0.1, 0.3], [0.5, 0.7]])  # song position [start_frac, end_frac]

  with torch.no_grad():
    logits = model(
      src=src,
      tgt_tokens=tgt,
      conditioning=conditioning,
    )

  # Output shape: (B, T_tgt, vocab_size)
  assert logits.shape == (B, T_tgt, vocab_size), \
    f"Expected ({B}, {T_tgt}, {vocab_size}), got {logits.shape}"


def test_model_conditioning_dim():
  """Encoder accepts mel + conditioning input and produces valid output."""
  model, tokenizer, config = _make_small_model()

  B, T_mel = 1, 50

  src = torch.randn(B, T_mel, config.audio.n_mels)
  conditioning = torch.tensor([[0.25, 0.50]])

  with torch.no_grad():
    memory, mask = model.encoder(src, None, conditioning=conditioning)

  # Memory should have shape (B, T_sub, d_model) where T_sub <= T_mel (due to subsampling)
  assert memory.dim() == 3
  assert memory.shape[0] == B
  assert memory.shape[2] == config.model.encoder.d_model
  # With Conv2dSubsampling, time dimension is reduced by ~4x
  assert memory.shape[1] > 0
  assert memory.shape[1] <= T_mel


def test_model_forward_without_conditioning():
  """Forward pass without conditioning still works (backward compat)."""
  model, tokenizer, config = _make_small_model()
  vocab_size = len(tokenizer.vocab)

  B, T_mel, T_tgt = 1, 80, 15

  src = torch.randn(B, T_mel, config.audio.n_mels)
  tgt = torch.randint(0, vocab_size, (B, T_tgt))

  with torch.no_grad():
    logits = model(
      src=src,
      tgt_tokens=tgt,
      conditioning=None,
    )

  assert logits.shape == (B, T_tgt, vocab_size)


def test_kv_cache_generation():
  """Step-by-step KV cache decoding produces logits of correct shape at each step."""
  model, tokenizer, config = _make_small_model()
  vocab_size = len(tokenizer.vocab)

  B, T_mel = 1, 60
  max_gen_len = 10

  src = torch.randn(B, T_mel, config.audio.n_mels)
  conditioning = torch.tensor([[0.0, 0.2]])

  with torch.no_grad():
    memory, mem_mask = model.encoder(src, None, conditioning=conditioning)

  # Init KV cache
  cache = model.decoder.init_kv_cache(
    bsz=B,
    device=memory.device,
    max_len=max_gen_len + 1,
  )

  # Start with a random token
  current_token = torch.randint(0, vocab_size, (B,))

  with torch.no_grad():
    for step in range(max_gen_len):
      logits, cache = model.decoder.forward_step(
        current_token,
        memory,
        cache=cache,
        memory_key_padding_mask=mem_mask,
        position=step,
      )

      # Logits shape: (B, 1, vocab_size)
      assert logits.shape == (B, 1, vocab_size), \
        f"Step {step}: expected ({B}, 1, {vocab_size}), got {logits.shape}"

      # Sample next token
      current_token = logits[:, -1, :].argmax(dim=-1)

  # Cache length should match number of steps
  assert cache[0].length == max_gen_len, \
    f"Cache length {cache[0].length} != {max_gen_len}"


def test_model_gradient_flow():
  """Verify gradients flow through the model with conditioning."""
  model, tokenizer, config = _make_small_model()
  model.train()
  vocab_size = len(tokenizer.vocab)

  B, T_mel, T_tgt = 2, 80, 15

  src = torch.randn(B, T_mel, config.audio.n_mels)
  tgt = torch.randint(1, vocab_size, (B, T_tgt))  # avoid pad_id=0
  conditioning = torch.tensor([[0.1, 0.3], [0.6, 0.9]])

  logits = model(
    src=src,
    tgt_tokens=tgt,
    conditioning=conditioning,
  )

  # Compute a dummy loss
  loss = logits.sum()
  loss.backward()

  # Check gradients exist on encoder input projection (which takes conditioning)
  assert model.encoder.input_proj.weight.grad is not None, \
    "No gradient on encoder input_proj"
  assert model.encoder.input_proj.weight.grad.abs().sum() > 0, \
    "Gradient on encoder input_proj is all zeros"
