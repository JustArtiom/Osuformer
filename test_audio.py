import librosa
import soundfile as sf
import os
from src.audio import audio_to_mel_spectrogram

spec = audio_to_mel_spectrogram("dataset/raw/2059503 Fly Project - Toca Toca (Sped Up & Cut Ver)/audio.ogg", sr=22050, n_fft=2048, hop_ms=.1, n_mels=128)

spec.show(figsize=(20, 8))