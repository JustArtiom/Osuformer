# test_freq.py
# Band-isolate a mel spectrogram between MIN_FREQ and MAX_FREQ and reconstruct audio.

import os
import numpy as np
import librosa
import soundfile as sf
from src.audio import audio_to_mel_spectrogram

# --------- EDIT THESE CONSTANTS ---------
INPUT_AUDIO   = "dataset/classes/jumps/2059503 Fly Project - Toca Toca (Sped Up & Cut Ver)/audio.ogg"
OUTPUT_AUDIO  = "./reconstructed_band.wav"
MIN_FREQ      = 000.0     # Hz (<= set this)
MAX_FREQ      = 1000.0    # Hz (>= set this)
N_ITER        = 64        # Griffin-Lim iterations
ATTEN_DB      = 120.0     # Attenuation for out-of-band bins (dB)
# ----------------------------------------

def mel_spec_to_audio(spec, *, n_iter: int = 64, length=None):
    """Invert a MelSpec back to waveform using stored analysis params."""
    M = librosa.db_to_power(spec.S_db, ref=spec.ref)
    y = librosa.feature.inverse.mel_to_audio(
        M,
        sr=spec.sr,
        n_fft=spec.n_fft,
        hop_length=spec.hop_length,
        win_length=None,
        window=spec.window,
        center=spec.center,
        pad_mode="reflect",
        power=spec.power,
        n_iter=n_iter,
        length=length,
        # Use the same mel config that created the spectrogram
        fmin=spec.fmin,
        fmax=spec.fmax,
    )
    return y

def main():
    # 1) Build mel spectrogram from input
    spec = audio_to_mel_spectrogram(
        INPUT_AUDIO,
        # Keep analysis wide; you can change these upstream if needed
        sr=22050,
        n_fft=2048,
        hop_ms=10.0,
        n_mels=128,
        fmin=30.0,
        fmax=None,   # Use Nyquist
        power=2.0,
        ref=1.0,
        top_db=80.0,
        mono=True,
    )

    # 2) Build a mask over mel center frequencies
    freqs = spec.freqs  # Hz, shape: (n_mels,)
    if MIN_FREQ >= MAX_FREQ:
        raise ValueError(f"MIN_FREQ ({MIN_FREQ}) must be < MAX_FREQ ({MAX_FREQ}).")
    in_band = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)

    if not np.any(in_band):
        raise ValueError(
            f"No mel bins within [{MIN_FREQ}, {MAX_FREQ}] Hz. "
            f"Valid mel band range is roughly [{freqs.min():.1f}, {freqs.max():.1f}] Hz."
        )

    # 3) Zero-out (strongly attenuate) mel bins outside the band in dB domain
    S_db_masked = np.array(spec.S_db, copy=True)
    out_of_band = ~in_band
    if np.any(out_of_band):
        min_val = np.min(S_db_masked)
        # Push out-of-band bins well below any in-band energy
        S_db_masked[out_of_band, :] = min(min_val, -ATTEN_DB)

    # 4) Temporarily swap the spectrogram for inversion
    S_db_orig = spec.S_db
    try:
        spec.S_db = S_db_masked
        y = mel_spec_to_audio(spec, n_iter=N_ITER)
    finally:
        spec.S_db = S_db_orig  # restore

    # 5) Save result
    sf.write(OUTPUT_AUDIO, y, spec.sr, subtype="PCM_16")
    print(f"Reconstructed band-limited audio saved to: {os.path.abspath(OUTPUT_AUDIO)}")
    print(f"Band: {MIN_FREQ}–{MAX_FREQ} Hz, sr={spec.sr}, hop_length={spec.hop_length}, n_fft={spec.n_fft}, n_mels={spec.n_mels}")

if __name__ == "__main__":
    main()