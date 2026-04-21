import torch

from src.model.attention import MultiHeadAttention


def _fixture_attn(d_model: int = 64, num_heads: int = 4, seed: int = 42) -> MultiHeadAttention:
    torch.manual_seed(seed)
    attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0).eval()
    return attn


def test_output_shape_matches_query_shape() -> None:
    attn = _fixture_attn()
    q = torch.randn(2, 16, 64)
    kv = torch.randn(2, 32, 64)
    out = attn(q, kv, kv, is_causal=False)
    assert out.shape == q.shape


def test_causal_mask_blocks_future_positions() -> None:
    attn = _fixture_attn()
    x = torch.randn(2, 16, 64)
    out = attn(x, x, x, is_causal=True)
    x_mod = x.clone()
    x_mod[:, 5:] *= 100
    out_mod = attn(x_mod, x_mod, x_mod, is_causal=True)
    assert (out[:, 4] - out_mod[:, 4]).abs().max().item() < 1e-4


def test_causal_mask_allows_past_attention() -> None:
    attn = _fixture_attn()
    x = torch.randn(2, 16, 64)
    out = attn(x, x, x, is_causal=True)
    x_mod = x.clone()
    x_mod[:, 5:] *= 100
    out_mod = attn(x_mod, x_mod, x_mod, is_causal=True)
    assert (out[:, 10] - out_mod[:, 10]).abs().max().item() > 0.01


def test_key_padding_mask_suppresses_padded_positions() -> None:
    attn = _fixture_attn()
    q = torch.randn(2, 8, 64)
    kv = torch.randn(2, 32, 64)
    pad = torch.zeros(2, 32, dtype=torch.bool)
    pad[0, 20:] = True
    out = attn(q, kv, kv, key_padding_mask=pad, is_causal=False)
    kv_mod = kv.clone()
    kv_mod[0, 20:] *= 100
    out_mod = attn(q, kv_mod, kv_mod, key_padding_mask=pad, is_causal=False)
    assert (out[0] - out_mod[0]).abs().max().item() < 1e-4


def test_key_padding_mask_preserves_unpadded_influence() -> None:
    attn = _fixture_attn()
    q = torch.randn(2, 8, 64)
    kv = torch.randn(2, 32, 64)
    pad = torch.zeros(2, 32, dtype=torch.bool)
    pad[0, 20:] = True
    out = attn(q, kv, kv, key_padding_mask=pad, is_causal=False)
    kv_mod = kv.clone()
    kv_mod[0, :10] *= 100
    out_mod = attn(q, kv_mod, kv_mod, key_padding_mask=pad, is_causal=False)
    assert (out[0] - out_mod[0]).abs().max().item() > 0.01


def test_combined_causal_and_padding_mask() -> None:
    attn = _fixture_attn()
    x = torch.randn(2, 16, 64)
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[1, 12:] = True
    out = attn(x, x, x, key_padding_mask=pad, is_causal=True)
    assert out.shape == x.shape
    x_mod = x.clone()
    x_mod[1, 12:] *= 100
    out_mod = attn(x_mod, x_mod, x_mod, key_padding_mask=pad, is_causal=True)
    assert (out[1, 2] - out_mod[1, 2]).abs().max().item() < 1e-4


def test_backward_produces_finite_gradients() -> None:
    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=64, num_heads=4, dropout=0.1).train()
    x = torch.randn(2, 16, 64, requires_grad=True)
    loss = attn(x, x, x, is_causal=True).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_dropout_disabled_in_eval_mode() -> None:
    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=64, num_heads=4, dropout=0.5).eval()
    x = torch.randn(2, 16, 64)
    out_a = attn(x, x, x, is_causal=True)
    out_b = attn(x, x, x, is_causal=True)
    assert torch.allclose(out_a, out_b)
