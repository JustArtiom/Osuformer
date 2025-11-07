import os

from src.utils.config import load_config
from src.utils.file import collect_files
from src.audio import audio_to_mel_spectrogram, AUDIO_EXTENSIONS, prepare_audio
from tqdm import tqdm

config = load_config("./config.yaml")

DATA_DIR = config['paths']['data']
EXCLUDE_PATTERNS = config['paths']['exclude']

print(f"Working directory: {DATA_DIR}")
print(f"Include patterns: {AUDIO_EXTENSIONS}")
print(f"Exclude patterns: {EXCLUDE_PATTERNS}")

files = collect_files(DATA_DIR, AUDIO_EXTENSIONS, EXCLUDE_PATTERNS)
if __name__ == "__main__":
    for f in tqdm(files, desc="Audio → Spectrogram", unit="file"):
        try:
            prepare_audio(DATA_DIR + "/" + f, config["data"]["sample_rate"], config["data"]["hop_ms"], config["data"]["win_ms"], config["data"]["n_mels"], config["data"]["n_fft"], force=True)
        except Exception as e:
            print(f"[ERROR] Failed processing {f}: {e}")