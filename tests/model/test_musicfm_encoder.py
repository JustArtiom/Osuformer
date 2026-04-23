import pytest
import torch

from src.config.schemas.audio import AudioConfig
from src.config.schemas.model import EncoderConfig
from src.model.encoders import build_audio_encoder
from src.model.encoders.musicfm import MusicFMEncoder


def _audio_cfg(n_mels: int = 128) -> AudioConfig:
    return AudioConfig(
        version="v3-musicfm",
        sample_rate=24000,
        hop_ms=10,
        win_ms=85,
        n_mels=n_mels,
        n_fft=2048,
        context_ms=5000,
        generate_ms=10000,
        lookahead_ms=5000,
        preset="musicfm",
        stats_path=None,
    )


def _encoder_cfg(layer: int = 12, freeze_first_n_layers: int = 0) -> EncoderConfig:
    return EncoderConfig(
        d_model=1024,
        num_heads=16,
        num_layers=12,
        ffn_dim=4096,
        conv_kernel=31,
        dropout=0.0,
        type="musicfm",
        musicfm_layer=layer,
        freeze_first_n_layers=freeze_first_n_layers,
    )


@pytest.fixture(scope="module")
def musicfm_encoder() -> MusicFMEncoder:
    return MusicFMEncoder(_encoder_cfg(), _audio_cfg())


def test_musicfm_encoder_output_shape(musicfm_encoder: MusicFMEncoder) -> None:
    mel = torch.randn(2, 200, 128)
    with torch.no_grad():
        out = musicfm_encoder(mel)
    assert out.shape == (2, 50, 1024)
    assert musicfm_encoder.output_dim == 1024
    assert musicfm_encoder.feature_rate_hz == 25.0


def test_musicfm_encoder_layer_selection_changes_output() -> None:
    enc_a = MusicFMEncoder(_encoder_cfg(layer=6), _audio_cfg())
    enc_b = MusicFMEncoder(_encoder_cfg(layer=12), _audio_cfg())
    enc_b.load_state_dict(enc_a.state_dict())
    mel = torch.randn(1, 100, 128)
    with torch.no_grad():
        out_a = enc_a(mel)
        out_b = enc_b(mel)
    assert out_a.shape == out_b.shape
    assert not torch.allclose(out_a, out_b)


def test_musicfm_encoder_freeze_reduces_trainable_params() -> None:
    full = MusicFMEncoder(_encoder_cfg(freeze_first_n_layers=0), _audio_cfg())
    frozen = MusicFMEncoder(_encoder_cfg(freeze_first_n_layers=4), _audio_cfg())
    n_full = sum(p.numel() for p in full.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    assert n_frozen < n_full


def test_musicfm_encoder_rejects_wrong_n_mels() -> None:
    with pytest.raises(ValueError, match="n_mels=128"):
        MusicFMEncoder(_encoder_cfg(), _audio_cfg(n_mels=80))


def test_musicfm_encoder_rejects_invalid_layer() -> None:
    with pytest.raises(ValueError, match="musicfm_layer"):
        MusicFMEncoder(_encoder_cfg(layer=42), _audio_cfg())


def test_factory_dispatches_to_musicfm() -> None:
    enc = build_audio_encoder(_encoder_cfg(), _audio_cfg(), max_len=500)
    assert isinstance(enc, MusicFMEncoder)


def test_factory_dispatches_to_conformer_scratch_by_default() -> None:
    from src.model.encoders.conformer_scratch import ConformerScratchEncoder

    encoder_cfg = EncoderConfig(d_model=64, num_heads=4, num_layers=2, ffn_dim=128, conv_kernel=31, dropout=0.0)
    audio_cfg = AudioConfig(
        version="v1", sample_rate=22050, hop_ms=10, win_ms=25, n_mels=128, n_fft=1024,
        context_ms=5000, generate_ms=10000, lookahead_ms=5000,
    )
    enc = build_audio_encoder(encoder_cfg, audio_cfg, max_len=500)
    assert isinstance(enc, ConformerScratchEncoder)


def test_factory_rejects_unknown_type() -> None:
    encoder_cfg = EncoderConfig(
        d_model=64, num_heads=4, num_layers=2, ffn_dim=128, conv_kernel=31, dropout=0.0, type="bogus",
    )
    audio_cfg = AudioConfig(
        version="v1", sample_rate=22050, hop_ms=10, win_ms=25, n_mels=128, n_fft=1024,
        context_ms=5000, generate_ms=10000, lookahead_ms=5000,
    )
    with pytest.raises(ValueError, match="bogus"):
        build_audio_encoder(encoder_cfg, audio_cfg, max_len=500)
