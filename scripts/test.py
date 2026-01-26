from src.data.audio import audio_to_mel
from src.config import ExperimentConfig
import librosa

config = ExperimentConfig()
mel_db = audio_to_mel("./polar240.mp3", 
  **config.audio.__dict__ 
)[0]

print(mel_db.shape)

# save mel to image
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, 
  sr=config.audio.sample_rate, 
  hop_length=(config.audio.hop_ms * config.audio.sample_rate) // 1000,
  win_length=(config.audio.win_ms * config.audio.sample_rate) // 1000,
  n_fft=config.audio.n_fft,
  x_axis='time', y_axis='mel'
)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.savefig("mel_spectrogram.png")
print("Saved mel spectrogram to mel_spectrogram.png")