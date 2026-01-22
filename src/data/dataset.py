import os
from tqdm import tqdm
from typing import List, Union
from pathlib import Path
import random
import numpy as np
from multiprocessing import Pool, cpu_count

from .audio import audio_to_mel, compute_mel_stats, normalize_mel, StreamingAudioStats
from ..osu import Beatmap
from .path import mkdir
from .crypt import file_hash
from ..config import ExperimentConfig
from ..tokenizer import Tokenizer
from .analytics import Analytics

class Dataset:
  def __init__(self, config: ExperimentConfig, workers: int | None = None):
    self.tokenizer = Tokenizer(config.tokenizer)
    self.config = config
    self.workers = workers or config.dataset.workers

  def get_osu_files(self) -> List[Path]:
    return list(tqdm(
      Path(self.config.dataset.path).rglob("*.osu"),
      desc="Scanning osu files"
    ))

  def build_cache(self, name: str, limit: int = -1):
    if not name:
      raise ValueError("Cache name must be provided")

    print("Preparing cache directory...")
    pwd = mkdir(Path(self.config.cache.path) / name, overwrite=True)
    analytics = Analytics(parent_path=pwd / "analytics")
    audio_pwd = mkdir(pwd / "audio")
    map_pwd = mkdir(pwd / "maps")
    print(f"Cache directory: {pwd}")

    print("Saving tokenizer vocabulary...")
    npz_vocab = pwd / f"vocab.json"
    self.tokenizer.save_vocab(str(npz_vocab))

    print("Listing osu! files...")
    osu_files = self.get_osu_files()
    random.shuffle(osu_files)
    if limit > 0:
      print(f"Limiting to {limit} files...")
      osu_files = osu_files[:limit]

    split = self.split_dataset(osu_files)
    audioStats = StreamingAudioStats()
    tokenizer = Tokenizer(self.config.tokenizer)


    for type_split, files in split.items():
      jobs = [(p, self.config, pwd, tokenizer) for p in files]
      save_i = 0
      type_split_pwd = mkdir(map_pwd / type_split)
      with Pool(min(self.workers, cpu_count())) as pool:
        for (beatmap, tokens, hash_id, audio_mel, duration_ms) in tqdm(
          pool.imap_unordered(process_map_sample, jobs),
          total=len(jobs),
          desc=f"Processing {type_split} cache"
        ):
          if hash_id is None or audio_mel is None or tokens is None or beatmap is None or duration_ms is None:
            continue
          np.savez_compressed(
            type_split_pwd / f"{save_i:08d}.npz",
            audio_id=hash_id,
            tokens=tokens
          )
          analytics.collect_beatmap(beatmap)

          if not os.path.exists(audio_pwd / f"{hash_id}.npz"):
            audioStats.update(audio_mel)
            np.savez_compressed(
              audio_pwd / f"{hash_id}.npz",
              mel=audio_mel
            )
            analytics.collect_audio(
              duration_ms=duration_ms,
              mel=audio_mel
            )
          save_i += 1


    print("Computing audio statistics...")
    mean, std = audioStats.finalize()
    print(f"Mel spectrogram mean: {mean}, std: {std}")
    audioStats.save(pwd / "mel_stats.json")

    analytics.save()

  def get_split_ratios(self, amount: int, ratios: List[float]) -> List[int]:
    total = sum(ratios)
    splits = [int((r / total) * amount) for r in ratios]
    diff = amount - sum(splits)
    for i in range(diff):
      splits[i % len(splits)] += 1
    return splits
  
  def split_dataset(self, files: List[Path]) -> dict[str, List[Path]]:
    count_train, count_val = self.get_split_ratios(len(files), [
      self.config.dataset.split.train,
      self.config.dataset.split.val
    ])
    
    return {
      "train": files[:count_train],
      "val": files[count_train:count_train + count_val]
    }
  

def process_map_sample(args: tuple[Path, ExperimentConfig, Path, Tokenizer]):
  osu_path, config, pwd, tokenizer = args
  if Beatmap.get_mode(str(osu_path)) != 0:
    return None, None, None, None, None

  beatmap = Beatmap(file_path=str(osu_path))
  tokens = tokenizer.encode(beatmap)

  audio_path = osu_path.parent / beatmap.general.audio_filename
  hash_id = file_hash(audio_path)
  cache_audio_file = Path(pwd) / "audio" / f"{hash_id}.npz"
  
  if cache_audio_file.exists():
    return None, None, None, None, None

  audio_mel, duration_ms = audio_to_mel(
    path=audio_path,
    sample_rate=config.audio.sample_rate,
    hop_ms=config.audio.hop_ms,
    win_ms=config.audio.win_ms,
    n_mels=config.audio.n_mels,
    n_fft=config.audio.n_fft,
  )

  return beatmap, tokens, hash_id, audio_mel, duration_ms