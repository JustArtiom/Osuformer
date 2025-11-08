import os
from concurrent.futures import ThreadPoolExecutor

from src.utils.config import load_config
from src.utils.file import collect_files
from src.audio import AUDIO_EXTENSIONS, prepare_audio
from tqdm import tqdm

config = load_config("./config.yaml")

DATA_DIR = config['paths']['data']
EXCLUDE_PATTERNS = config['paths']['exclude']
DATA_CFG = config["data"]
PREP_WORKERS = max(1, int(DATA_CFG.get("prep_num_workers", 1)))

print(f"Working directory: {DATA_DIR}")
print(f"Include patterns: {AUDIO_EXTENSIONS}")
print(f"Exclude patterns: {EXCLUDE_PATTERNS}")
print(f"Prep workers: {PREP_WORKERS}")

files = collect_files(DATA_DIR, AUDIO_EXTENSIONS, EXCLUDE_PATTERNS)


def _prepare_single(rel_path: str):
    full_path = os.path.join(DATA_DIR, rel_path)
    try:
        prepare_audio(
            full_path,
            DATA_CFG["sample_rate"],
            DATA_CFG["hop_ms"],
            DATA_CFG["win_ms"],
            DATA_CFG["n_mels"],
            DATA_CFG["n_fft"],
            force=True,
        )
    except Exception as e:
        return rel_path, str(e)
    return rel_path, None


if __name__ == "__main__":
    if not files:
        print("[INFO] No audio files found to process.")
    elif PREP_WORKERS == 1:
        iterator = tqdm(files, desc="Audio → Spectrogram", unit="file")
        for rel_path in iterator:
            _, error = _prepare_single(rel_path)
            if error:
                print(f"[ERROR] Failed processing {rel_path}: {error}")
    else:
        with ThreadPoolExecutor(max_workers=PREP_WORKERS) as executor:
            results = executor.map(_prepare_single, files)
            if tqdm is not None:
                results = tqdm(results, total=len(files), desc="Audio → Spectrogram", unit="file")
            for rel_path, error in results:
                if error:
                    print(f"[ERROR] Failed processing {rel_path}: {error}")
