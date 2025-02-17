import numpy as np
import librosa

def load_audio(file_path: str, sample_rate: int, n_mels: int, frame_ms: int):
    y, sr = librosa.load(file_path, sr=sample_rate)
    frame_length = int((frame_ms / 1000) * sr)
    hop_length = frame_length

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=frame_length)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram.T, sr