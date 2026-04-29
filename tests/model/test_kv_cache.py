import torch

from src.config.schemas.model import DecoderConfig
from src.model.decoder import TransformerDecoder


def _fixture_decoder(
    vocab_size: int = 64,
    d_model: int = 32,
    num_heads: int = 4,
    num_layers: int = 3,
    max_len: int = 128,
    cond_dim: int = 32,
    seed: int = 0,
) -> TransformerDecoder:
    torch.manual_seed(seed)
    cfg = DecoderConfig(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=d_model * 4,
        dropout=0.0,
    )
    return TransformerDecoder(config=cfg, vocab_size_in=vocab_size, max_len=max_len, cond_dim=cond_dim).eval()


def _zero_cond(batch: int, cond_dim: int = 32) -> torch.Tensor:
    return torch.zeros(batch, cond_dim)


def test_step_matches_forward_full_sequence() -> None:
    decoder = _fixture_decoder()
    torch.manual_seed(1)
    memory = torch.randn(1, 20, 32)
    input_ids = torch.randint(0, 64, (1, 12))
    cond = _zero_cond(1)

    full_out = decoder(input_ids, memory=memory, cond=cond)

    cache = None
    step_outputs = []
    for t in range(input_ids.shape[1]):
        step_in = input_ids[:, t : t + 1]
        out, cache = decoder.step(step_in, memory=memory, cond=cond, cache=cache, start_pos=t)
        step_outputs.append(out)
    incremental = torch.cat(step_outputs, dim=1)

    assert incremental.shape == full_out.shape
    assert torch.allclose(incremental, full_out, atol=1e-5, rtol=1e-5)


def test_step_after_prompt_matches_full() -> None:
    decoder = _fixture_decoder()
    torch.manual_seed(2)
    memory = torch.randn(1, 16, 32)
    input_ids = torch.randint(0, 64, (1, 20))
    cond = _zero_cond(1)

    full_out = decoder(input_ids, memory=memory, cond=cond)

    prompt = input_ids[:, :8]
    prompt_out, cache = decoder.step(prompt, memory=memory, cond=cond, cache=None, start_pos=0)
    assert torch.allclose(prompt_out, full_out[:, :8], atol=1e-5, rtol=1e-5)

    per_step = [prompt_out]
    for t in range(8, input_ids.shape[1]):
        step_in = input_ids[:, t : t + 1]
        out, cache = decoder.step(step_in, memory=memory, cond=cond, cache=cache, start_pos=t)
        per_step.append(out)
    incremental = torch.cat(per_step, dim=1)
    assert torch.allclose(incremental, full_out, atol=1e-5, rtol=1e-5)


def test_cross_kv_cached_across_steps() -> None:
    decoder = _fixture_decoder()
    torch.manual_seed(3)
    memory = torch.randn(1, 12, 32)
    input_ids = torch.randint(0, 64, (1, 6))
    cond = _zero_cond(1)

    cache = None
    for t in range(input_ids.shape[1]):
        step_in = input_ids[:, t : t + 1]
        _, cache = decoder.step(step_in, memory=memory, cond=cond, cache=cache, start_pos=t)

    assert cache is not None
    for block_cache in cache:
        assert block_cache.cross_kv is not None
        ck, cv = block_cache.cross_kv
        assert ck.shape[2] == memory.shape[1]
        assert cv.shape[2] == memory.shape[1]


def test_self_kv_grows_with_each_step() -> None:
    decoder = _fixture_decoder()
    torch.manual_seed(4)
    memory = torch.randn(1, 8, 32)
    input_ids = torch.randint(0, 64, (1, 5))
    cond = _zero_cond(1)

    cache = None
    for t in range(input_ids.shape[1]):
        step_in = input_ids[:, t : t + 1]
        _, cache = decoder.step(step_in, memory=memory, cond=cond, cache=cache, start_pos=t)
        for block_cache in cache:
            assert block_cache.self_kv is not None
            sk, sv = block_cache.self_kv
            assert sk.shape[2] == t + 1
            assert sv.shape[2] == t + 1
