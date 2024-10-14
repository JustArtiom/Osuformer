import librosa
import numpy as np


def process_audio_file(file_path: str, sample_rate=22050, n_mfcc=13, n_mels=128, hop_length=512):
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    duration_ms = librosa.get_duration(y=audio, sr=sr) * 1000
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_db = (mel_spectrogram_db - np.min(mel_spectrogram_db)) / \
                         (np.max(mel_spectrogram_db) - np.min(mel_spectrogram_db))

    return mel_spectrogram_db, duration_ms


def preprocess_audio_files(audio_file_paths, sample_rate=22050, n_mfcc=13, n_mels=128, hop_length=512):
    preprocessed_data = []
    audio_duration_ms = []

    for file_path in audio_file_paths:
        proc, duration_ms_file = process_audio_file(
            file_path, sample_rate, n_mfcc, n_mels, hop_length)
        preprocessed_data.append(proc)
        audio_duration_ms.append(duration_ms_file)

    return preprocessed_data, audio_duration_ms
