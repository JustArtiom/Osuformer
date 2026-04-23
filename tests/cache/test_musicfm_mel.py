import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.cache.audio import compute_mel
from src.config.schemas.audio import AudioConfig


def _write_test_audio(path: Path, duration_s: float = 2.0, sr: int = 22050) -> None:
    rng = np.random.default_rng(seed=0)
    sig = rng.standard_normal(int(sr * duration_s)).astype(np.float32) * 0.1
    sf.write(str(path), sig, sr)


def _write_stats(path: Path, mean: float = -30.0, std: float = 20.0) -> None:
    with open(path, "w") as f:
        json.dump({"melspec_2048_mean": mean, "melspec_2048_std": std}, f)


def _musicfm_audio_cfg(stats_path: str | None) -> AudioConfig:
    return AudioConfig(
        version="v3-musicfm",
        sample_rate=24000,
        hop_ms=10,
        win_ms=85,
        n_mels=128,
        n_fft=2048,
        context_ms=5000,
        generate_ms=10000,
        lookahead_ms=5000,
        preset="musicfm",
        stats_path=stats_path,
    )


def test_musicfm_mel_shape_and_dtype(tmp_path: Path) -> None:
    wav = tmp_path / "test.wav"
    stats = tmp_path / "msd_stats.json"
    _write_test_audio(wav, duration_s=2.0)
    _write_stats(stats)
    cfg = _musicfm_audio_cfg(stats_path=str(stats))
    mel = compute_mel(wav, cfg)
    assert mel.dtype == np.float16
    assert mel.ndim == 2
    assert mel.shape[1] == 128
    expected_frames = int(2.0 * cfg.sample_rate / int(cfg.hop_ms * cfg.sample_rate / 1000)) - 1
    assert abs(mel.shape[0] - expected_frames) <= 1


def test_musicfm_mel_normalization_changes_with_stats(tmp_path: Path) -> None:
    wav = tmp_path / "test.wav"
    _write_test_audio(wav, duration_s=1.0)
    stats_a = tmp_path / "stats_a.json"
    stats_b = tmp_path / "stats_b.json"
    _write_stats(stats_a, mean=-30.0, std=20.0)
    _write_stats(stats_b, mean=-50.0, std=10.0)
    mel_a = compute_mel(wav, _musicfm_audio_cfg(str(stats_a))).astype(np.float32)
    mel_b = compute_mel(wav, _musicfm_audio_cfg(str(stats_b))).astype(np.float32)
    assert mel_a.shape == mel_b.shape
    assert not np.allclose(mel_a, mel_b)


def test_musicfm_preset_requires_stats_path(tmp_path: Path) -> None:
    wav = tmp_path / "test.wav"
    _write_test_audio(wav)
    cfg = _musicfm_audio_cfg(stats_path=None)
    with pytest.raises(ValueError, match="stats_path"):
        compute_mel(wav, cfg)


def test_default_preset_still_works(tmp_path: Path) -> None:
    wav = tmp_path / "test.wav"
    _write_test_audio(wav, duration_s=1.0)
    cfg = AudioConfig(
        version="v1", sample_rate=22050, hop_ms=10, win_ms=25,
        n_mels=128, n_fft=1024, context_ms=5000, generate_ms=10000, lookahead_ms=5000,
    )
    mel = compute_mel(wav, cfg)
    assert mel.dtype == np.float16
    assert mel.shape[1] == 128


def test_unknown_preset_raises() -> None:
    cfg = AudioConfig(
        version="bogus", sample_rate=22050, hop_ms=10, win_ms=25,
        n_mels=128, n_fft=1024, context_ms=5000, generate_ms=10000, lookahead_ms=5000,
        preset="bogus",
    )
    with pytest.raises(ValueError, match="bogus"):
        compute_mel(Path("/dev/null"), cfg)
