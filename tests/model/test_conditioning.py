import torch

from src.config.loader import load_config
from src.model.conditioning import (
    ConditionEncoder,
    ConditionFeatures,
    default_condition_spec,
    encode_condition_features,
)


def _spec(cond_dim: int = 32):
    cfg = load_config("config/config.yaml").tokenizer
    spec = default_condition_spec(cfg, cond_dim=cond_dim)
    return cfg, spec


def _features(spec, batch: int = 2) -> ConditionFeatures:
    return ConditionFeatures(
        scalars=torch.randn(batch, 8),
        year_idx=torch.zeros(batch, dtype=torch.long),
        descriptors=torch.zeros(batch, spec.descriptor_count),
    )


def test_cond_encoder_outputs_correct_shape() -> None:
    _, spec = _spec(cond_dim=64)
    encoder = ConditionEncoder(spec).eval()
    out = encoder(_features(spec, batch=3))
    assert out.shape == (3, 64)


def test_null_mask_replaces_with_null_embedding() -> None:
    _, spec = _spec(cond_dim=32)
    encoder = ConditionEncoder(spec).eval()
    feats = _features(spec, batch=4)
    full = encoder(feats)
    mask = torch.tensor([True, False, True, False])
    masked = encoder(feats, null_mask=mask)
    null = encoder.null_embedding
    assert torch.allclose(masked[0], null)
    assert torch.allclose(masked[2], null)
    assert torch.allclose(masked[1], full[1])
    assert torch.allclose(masked[3], full[3])


def test_null_vector_returns_broadcast_null() -> None:
    _, spec = _spec(cond_dim=24)
    encoder = ConditionEncoder(spec).eval()
    null = encoder.null_vector(batch_size=5, device=torch.device("cpu"))
    assert null.shape == (5, 24)
    for i in range(5):
        assert torch.allclose(null[i], encoder.null_embedding)


def test_encode_condition_features_handles_missing_metadata() -> None:
    cfg, spec = _spec()
    map_record = {
        "hitsounded": True,
        "circle_size": 4.0,
        "approach_rate": 9.0,
        "overall_difficulty": 8.0,
        "hp_drain_rate": 6.0,
        "slider_multiplier": 1.4,
        "duration_ms": 5000.0,
    }
    feats = encode_condition_features(
        map_record=map_record,
        metadata=None,
        tokenizer_cfg=cfg,
        descriptor_count=spec.descriptor_count,
    )
    assert feats.scalars.shape == (8,)
    assert feats.year_idx.item() == 0
    assert feats.descriptors.sum().item() == 0.0
