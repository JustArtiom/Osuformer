from audio import audio_to_mel_spectrogram
import soundfile as sf

spec = audio_to_mel_spectrogram("dataset/audio.mp3", sr=22050, n_fft=2048, hop_ms=1, n_mels=128)
# spec.show(figsize=(20, 8))

import numpy as np
import librosa
import soundfile as sf  # pip install soundfile

def mel_spec_to_audio(
    spec,
    *,
    n_iter: int = 32,         # Griffin–Lim iterations (↑ = better, slower)
    use_builtin: bool = True, # True -> librosa.feature.inverse.mel_to_audio
    out_path: str | None = None,
    normalize: bool = True,   # peak normalize output before saving
) -> np.ndarray:
    """
    Invert a log-mel (dB) spectrogram (MelSpec) back to audio.
    Returns the waveform as float32; optionally writes a WAV.

    NOTE: This is lossy (mel compression) + iterative (phase estimation).
    """
    sr = spec.sr
    n_fft = spec.n_fft
    hop_length = spec.hop_length
    win_length = None  # use librosa default (= n_fft)
    window = spec.window
    center = spec.center

    # 1) dB -> power mel
    S_mel = librosa.db_to_power(spec.S_db, ref=spec.ref)

    if use_builtin:
        # 2) mel -> audio (internally mel_to_stft + griffinlim)
        y = librosa.feature.inverse.mel_to_audio(
            S_mel,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            n_iter=n_iter,
            fmin=spec.fmin,
            fmax=spec.fmax,
            power=spec.power,
        )

    else:
        # 2a) mel -> linear STFT magnitude
        S_lin = librosa.feature.inverse.mel_to_stft(
            S_mel,
            sr=sr,
            n_fft=n_fft,
            fmin=spec.fmin,
            fmax=spec.fmax,
            power=spec.power,
            n_mels=spec.n_mels,
        )
        # 2b) Griffin–Lim to estimate phase & invert
        y = librosa.griffinlim(
            S_lin,
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
        )

    # Optional peak normalization for a sensible listening level
    if normalize and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    if out_path:
        sf.write(out_path, y.astype("float32"), sr)

    return y.astype("float32")

y = mel_spec_to_audio(spec, n_iter=32)
sf.write("reconstructed.wav", y, spec.sr)

print("✅ Saved: spectrogram.png & reconstructed.wav")