import torch
import os
from tqdm import tqdm
from typing import Dict, List, Literal, Union, Tuple
from pathlib import Path
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset as TorchDataset

from .audio import audio_to_mel, normalize_mel, compute_mel_stats, StreamingAudioStats, ms_to_samples
from ..osu import Beatmap, Slider
from .path import mkdir
from .crypt import file_hash
from ..config import ExperimentConfig
from ..tokenizer import Tokenizer
from .analytics import DatasetAnalytics

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
    analytics = DatasetAnalytics(parent_path=pwd / "analytics")
    mkdir(pwd / "audio")
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
    for type_split, files in split.items():
      jobs = [(p, self.config, pwd, self.tokenizer) for p in files]
      save_i = 0
      type_split_pwd = mkdir(map_pwd / type_split)
      with Pool(min(self.workers, cpu_count())) as pool:
        for (beatmap, tokens, times, hash_id, duration_ms, new_audio) in tqdm(
          pool.imap_unordered(process_map_sample, jobs),
          total=len(jobs),
          desc=f"Processing {type_split} cache"
        ):
          if hash_id is None or tokens is None or beatmap is None or duration_ms is None or times is None:
            continue

          np.savez_compressed(
            type_split_pwd / f"{save_i:08d}.npz",
            audio_id=hash_id,
            tokens=tokens,
            times=times
          )
          analytics.collect_beatmap(beatmap)

          if new_audio:
            analytics.collect_audio(
              duration_ms=duration_ms,
            )
          save_i += 1


    print("Computing audio statistics...")
    analytics.save()
    print("Cache build complete.")

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
    if not is_map_valid(beatmap, config):
      return None, None, None, None, None, None
    
    tokens, times = tokenizer.encode(beatmap)

    audio_path = osu_path.parent / beatmap.general.audio_filename
    hash_id = file_hash(audio_path)
    cache_audio_file = Path(pwd) / "audio" / f"{hash_id}.npz"

    new_audio = False
    if not cache_audio_file.exists():
      audio_mel, duration_ms = audio_to_mel(
        path=audio_path,
        sample_rate=config.audio.sample_rate,
        hop_ms=config.audio.hop_ms,
        win_ms=config.audio.win_ms,
        n_mels=config.audio.n_mels,
        n_fft=config.audio.n_fft,
      )
      new_audio = True
      np.savez_compressed(
        cache_audio_file,
        mel=audio_mel.T  # Transpose to (time, n_mels)
      )
    else:
      audio_npz = np.load(cache_audio_file, mmap_mode="r")
      duration_ms: float = audio_npz["mel"].shape[0] * config.audio.hop_ms

    return beatmap, tokens, times, hash_id, duration_ms, new_audio
  except Exception as e:
    print(f"Error processing {osu_path}: {e}")
    return None, None, None, None, None, None


def is_map_valid(
  beatmap: Beatmap,
  config: ExperimentConfig,
) -> bool:
  sr = beatmap.get_difficulty().star_rating
  if sr < config.dataset.map_filters.sr_min or sr > config.dataset.map_filters.sr_max:
    return False

  bpm_list = [tp.get_bpm() for tp in beatmap.timing_points if tp.uninherited]
  bpm_min = min(bpm_list) if bpm_list else 0
  bpm_max = max(bpm_list) if bpm_list else 0

  if bpm_list:
    if not config.dataset.map_filters.variable_bpm and bpm_min != bpm_max:
      return False
  else:
    return False
    
  if config.dataset.map_filters.tokenizer_limits:
    slider_cps = [len(ho.object_params.curves) for ho in beatmap.hit_objects if isinstance(ho, Slider)]
    if slider_cps:
      if max(slider_cps) > config.tokenizer.SLIDER_CP_LIMIT:
        return False
    
    svs = [
      tp.get_slider_velocity_multiplier() * beatmap.difficulty.slider_multiplier
      for tp in beatmap.timing_points if tp.uninherited
    ]
    if svs and max(svs) > config.tokenizer.SLIDER_VEL_LIMIT:
      return False
    
    if bpm_max > config.tokenizer.BPM_MAX or bpm_min < config.tokenizer.BPM_MIN:
      return False
    
    sl_lens = [ho.object_params.length for ho in beatmap.hit_objects if isinstance(ho, Slider)]
    max_sl_len = max(sl_lens) if sl_lens else 0
    if max_sl_len > config.tokenizer.SLIDER_LEN_MAX:
      return False
    
  return True

class CachedDataset(TorchDataset):
  def __init__(
    self, 
    parent_path: Path, 
    split: Literal["train", "val"],
    audioStats: StreamingAudioStats,
    window_ms: int,
    hop_ms: int,
    overlap: float,
    sample_rate: int | None = None,
    use_ram: bool = True,
  ):
    if not parent_path.exists():
      raise ValueError(f"Cache path does not exist: {parent_path}")
    self.map_files = sorted((parent_path / "maps" / split).glob("*.npz"))
    self.audio_dir = parent_path / "audio"
    self.vocab_path = parent_path / "vocab.json"
    self.tokenizer = Tokenizer().load(str(self.vocab_path))
    self.window_ms = window_ms
    self.split = split
    self.hop_ms = hop_ms
    self.overlap = overlap
    self.segment_frames = window_ms // hop_ms
    self.hop_frames = int(self.segment_frames * (1 - overlap))
    if sample_rate is None:
      self.hop_ms_actual = float(hop_ms)
    else:
      hop_samples = ms_to_samples(sample_rate, hop_ms)
      self.hop_ms_actual = hop_samples * 1000.0 / sample_rate
    self.use_ram = use_ram
    self.audioStats = audioStats
    self.token_window_builder = TokenWindowBuilder(
      tokenizer=self.tokenizer,
      max_tokens=1024, # TODO
      overlap_ratio=self.overlap,
    )

    self.frames = [] 
    self.mels: Dict[str, np.ndarray] = {}
    self.tokens: Dict[int, np.ndarray] = {}
    self.times: Dict[int, np.ndarray] = {}

    for map_idx, map_file in enumerate(tqdm(self.map_files, desc=f"[{self.split}]Preparing dataset frames", unit="maps", total=len(self.map_files))):
      map_npz = np.load(map_file, mmap_mode="r")
      audio_id = map_npz["audio_id"].item()
      if self.use_ram:
        self.tokens[map_idx] = map_npz["tokens"]
        self.times[map_idx] = map_npz["times"]

      if audio_id not in self.mels:
        audio_npz = np.load(self.audio_dir / f"{audio_id}.npz", mmap_mode="r")
        self.audioStats.update(audio_npz["mel"])
        if self.use_ram:
          self.mels[audio_id] = audio_npz["mel"]
        mel_len = (self.mels[audio_id] if audio_id in self.mels else audio_npz["mel"]).shape[0]
        for start in range(0, mel_len - self.segment_frames + 1, self.hop_frames):
          self.frames.append((map_idx, audio_id, start))

        final_start = max(0, mel_len - self.segment_frames)
        if len(self.frames) == 0 or self.frames[-1] != (map_idx, audio_id, final_start):
          self.frames.append((map_idx, audio_id, final_start))

  def load_audio_stats(self, mean, std):
    self.mean, self.std = mean, std

  def __len__(self):
    return len(self.frames)

  @staticmethod
  def collate_batch(batch, pad_id: int = 0):
    mels, tokens, loss_masks = zip(*batch)
    mels = torch.stack(mels, dim=0)

    lengths = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
    max_len: int = int(lengths.max().item())

    padded_tokens = torch.full(
      (len(tokens), max_len),
      fill_value=pad_id,
      dtype=torch.long,
    )

    padded_loss_mask = torch.zeros(
      (len(tokens), max_len),
      dtype=torch.bool,
    )

    for i, (t, lm) in enumerate(zip(tokens, loss_masks)):
      padded_tokens[i, : t.size(0)] = t
      padded_loss_mask[i, : lm.size(0)] = lm

    token_pad_mask = padded_tokens == pad_id

    return mels, padded_tokens, padded_loss_mask, token_pad_mask


  def __getitem__(self, idx):
    map_idx, audio_id, start = self.frames[idx]

    if self.use_ram:
      tokens = self.tokens[map_idx]
      times  = self.times[map_idx]
    else:
      map_npz = np.load(self.map_files[map_idx], mmap_mode="r")
      tokens = map_npz["tokens"]
      times  = map_npz["times"]

    if self.use_ram:
      mel = self.mels[audio_id]
    else:
      audio_npz = np.load(self.audio_dir / f"{audio_id}.npz", mmap_mode="r")
      mel = audio_npz["mel"]
      
    segment = mel[start : start + self.segment_frames]

    if segment.shape[0] < self.segment_frames:
      pad_len = self.segment_frames - segment.shape[0]
      segment = np.pad(
        segment,
        ((0, pad_len), (0, 0)),
        mode="constant",
        constant_values=0.0,
      )

    segment = normalize_mel(
      segment,
      self.mean,
      self.std
    )

    segment_start_ms = int(round(start * self.hop_ms_actual))
    segment_end_ms  = int(round(segment_start_ms + (self.segment_frames * self.hop_ms_actual)))

    window_tokens, loss_mask = self.token_window_builder.build(
      tokens=tokens,
      times=times,
      audio_start_ms=segment_start_ms,
      audio_end_ms=segment_end_ms,
    )

    return (
      torch.from_numpy(segment).float(),
      torch.from_numpy(window_tokens).long(),
      torch.from_numpy(loss_mask),
    )
  
class TokenWindowBuilder():
  def __init__(
    self,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_ratio: float,
  ):
    assert 0.0 < overlap_ratio < 1.0
    self.tokenizer = tokenizer
    self.max_tokens = max_tokens
    self.overlap_ratio = overlap_ratio

    self.MAP_START = tokenizer.token_to_id["MAP_START"]
    self.MAP_END   = tokenizer.token_to_id["MAP_END"]
    self.EOS       = tokenizer.token_to_id["EOS"]

  def build(
    self,
    *,
    tokens: np.ndarray,            # (N,)
    times: np.ndarray,             # (N,) ms
    audio_start_ms: int,
    audio_end_ms: int,
  ) -> Tuple[np.ndarray, np.ndarray]:

    window_ms = audio_end_ms - audio_start_ms
    predict_start_ms = audio_start_ms + int(window_ms * self.overlap_ratio)

    context_mask = times < predict_start_ms
    target_mask  = (times >= predict_start_ms) & (times < audio_end_ms)

    context_tokens = tokens[context_mask]
    target_tokens  = tokens[target_mask]

    context_tokens = self._strip_structure(context_tokens)
    target_tokens  = self._strip_structure(target_tokens)

    map_start_idx = np.where(tokens == self.tokenizer.token_to_id["MAP_START"])[0][0]
    global_tokens = tokens[:map_start_idx]

    # --- token budget ---
    reserved = (
      len(global_tokens)
      + 1
      + len(target_tokens)
    )

    if reserved > self.max_tokens:
      target_tokens = target_tokens[-(self.max_tokens - len(global_tokens) - 1):]
      reserved = len(global_tokens) + 1 + len(target_tokens)

    available_context = self.max_tokens - reserved

    if available_context > 0 and len(context_tokens) > available_context:
      context_tokens = context_tokens[-available_context:]
    elif available_context <= 0:
      context_tokens = np.empty((0,), dtype=np.int64)

    window_tokens = np.concatenate([
      global_tokens,
      np.array([self.MAP_START], dtype=np.int64),
      context_tokens,
      target_tokens,
    ])

    if self.MAP_END in target_tokens:
      window_tokens = np.concatenate([
        window_tokens,
        np.array([self.EOS], dtype=np.int64),
      ])

    loss_mask = np.zeros(len(window_tokens), dtype=bool)

    target_start = len(global_tokens) + 1 + len(context_tokens)
    loss_mask[target_start : target_start + len(target_tokens)] = True

    if window_tokens[-1] == self.EOS:
      loss_mask[-1] = True

    return window_tokens, loss_mask

  def _strip_structure(self, tokens: np.ndarray) -> np.ndarray:
    if tokens.size == 0:
      return tokens

    mask = (
      (tokens != self.MAP_START)
      & (tokens != self.MAP_END)
      & (tokens != self.EOS)
    )
    return tokens[mask]
