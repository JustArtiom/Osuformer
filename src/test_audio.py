from audio import audio_to_mel_spectrogram

spec = audio_to_mel_spectrogram("dataset/audio.mp3", sr=22050, n_fft=2048, hop_ms=1, n_mels=128)
spec.show(figsize=(20, 8))