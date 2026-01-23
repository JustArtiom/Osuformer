import torch
import os
from tqdm import tqdm
from typing import List, Literal, Union
from pathlib import Path
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset as TorchDataset

from .audio import audio_to_mel, normalize_mel, StreamingAudioStats
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
    self.tokenizer.save(str(npz_vocab))

    print("Listing osu! files...")
    osu_files = self.get_osu_files()
    random.shuffle(osu_files)
    if limit > 0:
      print(f"Limiting to {limit} files...")
      osu_files = osu_files[:limit]

    split = self.split_dataset(osu_files)
    audioStats = StreamingAudioStats()


    for type_split, files in split.items():
      jobs = [(p, self.config, pwd, self.tokenizer) for p in files]
      save_i = 0
      type_split_pwd = mkdir(map_pwd / type_split)
      with Pool(min(self.workers, cpu_count())) as pool:
        for (beatmap, tokens, times, hash_id, audio_mel, duration_ms) in tqdm(
          pool.imap_unordered(process_map_sample, jobs),
          total=len(jobs),
          desc=f"Processing {type_split} cache"
        ):
          if hash_id is None or audio_mel is None or tokens is None or beatmap is None or duration_ms is None or times is None:
            continue
          np.savez_compressed(
            type_split_pwd / f"{save_i:08d}.npz",
            audio_id=hash_id,
            tokens=tokens,
            times=times
          )
          analytics.collect_beatmap(beatmap)

          if not os.path.exists(audio_pwd / f"{hash_id}.npz"):
            audioStats.update(audio_mel)
            np.savez_compressed(
              audio_pwd / f"{hash_id}.npz",
              mel=audio_mel.T  # Transpose to (time, n_mels)
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
  try: 
    if Beatmap.get_mode(str(osu_path)) != 0:
      return None, None, None, None, None, None

    beatmap = Beatmap(file_path=str(osu_path))
    tokens, times = tokenizer.encode(beatmap)

    audio_path = osu_path.parent / beatmap.general.audio_filename
    hash_id = file_hash(audio_path)
    cache_audio_file = Path(pwd) / "audio" / f"{hash_id}.npz"
    
    if cache_audio_file.exists():
      return None, None, None, None, None, None

    audio_mel, duration_ms = audio_to_mel(
      path=audio_path,
      sample_rate=config.audio.sample_rate,
      hop_ms=config.audio.hop_ms,
      win_ms=config.audio.win_ms,
      n_mels=config.audio.n_mels,
      n_fft=config.audio.n_fft,
    )

    return beatmap, tokens, times, hash_id, audio_mel, duration_ms
  except Exception as e:
    print(f"Error processing {osu_path}: {e}")
    return None, None, None, None, None, None
  

class CachedDataset(TorchDataset):
  def __init__(
    self, 
    parent_path: Path, 
    split: Literal["train", "val"],
    window_ms: int,
    hop_ms: int,
    overlap: float,
  ):
    if not parent_path.exists():
      raise ValueError(f"Cache path does not exist: {parent_path}")
    self.map_files = sorted((parent_path / "maps" / split).glob("*.npz"))
    self.audio_dir = parent_path / "audio"
    self.vocab_path = parent_path / "vocab.json"
    self.tokenizer = Tokenizer().load(str(self.vocab_path))
    self.audio_stats = StreamingAudioStats().load(parent_path / "mel_stats.json")
    self.window_ms = window_ms
    self.hop_ms = hop_ms
    self.overlap = overlap
    self.segment_frames = window_ms // hop_ms
    self.hop_frames = int(self.segment_frames * (1 - overlap))

    self.frames = []  # list of (map_idx, audio_id, start)

    for map_idx, map_file in enumerate(tqdm(self.map_files, desc="Preparing dataset frames", unit="maps", total=len(self.map_files))):
      map_npz = np.load(map_file, mmap_mode="r")
      audio_id = map_npz["audio_id"].item()

      audio_npz = np.load(self.audio_dir / f"{audio_id}.npz", mmap_mode="r")
      mel_len = audio_npz["mel"].shape[0]

      for start in range(0, mel_len - self.segment_frames + 1, self.hop_frames):
        self.frames.append((map_idx, audio_id, start))

  def __len__(self):
    return len(self.frames)

  @staticmethod
  def collate_batch(batch, pad_id: int = 0):
    mels, tokens = zip(*batch)
    mels = torch.stack(mels, dim=0)

    lengths = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
    max_len: int = int(lengths.max().item())

    padded_tokens = torch.full(
        (len(tokens), max_len),
        fill_value=pad_id,
        dtype=torch.long,
    )

    for i, t in enumerate(tokens):
        padded_tokens[i, : t.size(0)] = t

    token_pad_mask = padded_tokens == pad_id

    return mels, padded_tokens, token_pad_mask


  def __getitem__(self, idx):
    map_idx, audio_id, start = self.frames[idx]

    map_npz = np.load(self.map_files[map_idx], mmap_mode="r")
    tokens = map_npz["tokens"]
    times  = map_npz["times"]

    audio_npz = np.load(self.audio_dir / f"{audio_id}.npz", mmap_mode="r")
    mel = audio_npz["mel"]

    segment = mel[start : start + self.segment_frames]
    segment = normalize_mel(
        segment,
        self.audio_stats.mean,
        self.audio_stats.std
    )

    segment_start_ms = start * self.hop_ms
    segment_end_ms   = segment_start_ms + self.window_ms

    mask = (times >= segment_start_ms) & (times < segment_end_ms)
    segment_tokens = tokens[mask]

    segment_tokens = np.concatenate([
        [self.tokenizer.token_to_id["BOS"]],
        segment_tokens,
        [self.tokenizer.token_to_id["EOS"]],
    ])

    return (
        torch.from_numpy(segment).float(),          # (T, n_mels)
        torch.from_numpy(segment_tokens).long(),    # (L,)
    )