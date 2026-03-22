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

  def build_cache(self, name: str, limit: int = -1, overwrite: bool = True):
    if not name:
      raise ValueError("Cache name must be provided")

    print("Preparing cache directory...")
    pwd = mkdir(Path(self.config.cache.path) / name, overwrite=overwrite)
    analytics = DatasetAnalytics(parent_path=pwd / "analytics")
    mkdir(pwd / "audio")
    map_pwd = mkdir(pwd / "maps", overwrite=True)
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
        for (beatmap, tokens, times, snaps, hash_id, duration_ms, new_audio) in tqdm(
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
            times=times,
            snaps=snaps,
            duration_ms=duration_ms,
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
      return None, None, None, None, None, None, None

    beatmap = Beatmap(file_path=str(osu_path))
    if not is_map_valid(beatmap, config):
      return None, None, None, None, None, None, None

    tokens, times, snaps = tokenizer.encode(beatmap)

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

    return beatmap, tokens, times, snaps, hash_id, duration_ms, new_audio
  except Exception as e:
    print(f"Error processing {osu_path}: {e}")
    return None, None, None, None, None, None, None


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
    slider_cps = [
      len(curve.curve_points)
      for ho in beatmap.hit_objects
      if isinstance(ho, Slider)
      for curve in ho.object_params.curves
    ]
    if slider_cps:
      if max(slider_cps) > config.tokenizer.SLIDER_CP_LIMIT:
        return False

    svs = [
      tp.get_slider_velocity_multiplier() * beatmap.difficulty.slider_multiplier
      for tp in beatmap.timing_points if not tp.uninherited
    ]
    if svs and max(svs) > config.tokenizer.SLIDER_VEL_LIMIT:
      return False

    if bpm_max > config.tokenizer.BPM_MAX or bpm_min < config.tokenizer.BPM_MIN:
      return False

    slides = [ho.object_params.slides for ho in beatmap.hit_objects if isinstance(ho, Slider)]
    max_slides = max(slides) if slides else 0
    if max_slides > config.tokenizer.SLIDES_MAX:
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
      max_tokens=1024,
      overlap_ratio=self.overlap,
    )

    self.frames = []
    self.mels: Dict[str, np.ndarray] = {}
    self.tokens: Dict[int, np.ndarray] = {}
    self.times: Dict[int, np.ndarray] = {}
    self.snaps: Dict[int, np.ndarray] = {}
    self.durations: Dict[int, float] = {}

    for map_idx, map_file in enumerate(tqdm(self.map_files, desc=f"[{self.split}]Preparing dataset frames", unit="maps", total=len(self.map_files))):
      map_npz = np.load(map_file, mmap_mode="r")
      audio_id = map_npz["audio_id"].item()
      if self.use_ram:
        self.tokens[map_idx] = map_npz["tokens"]
        self.times[map_idx] = map_npz["times"]
        self.snaps[map_idx] = map_npz["snaps"]
      if "duration_ms" in map_npz:
        self.durations[map_idx] = float(map_npz["duration_ms"])

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
    mels, tokens, loss_masks, song_positions = zip(*batch)
    mels = torch.stack(mels, dim=0)
    song_positions = torch.stack(song_positions, dim=0)

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

    return mels, padded_tokens, padded_loss_mask, token_pad_mask, song_positions


  def __getitem__(self, idx):
    map_idx, audio_id, start = self.frames[idx]

    if self.use_ram:
      tokens = self.tokens[map_idx]
      times  = self.times[map_idx]
      snaps  = self.snaps[map_idx]
    else:
      map_npz = np.load(self.map_files[map_idx], mmap_mode="r")
      tokens = map_npz["tokens"]
      times  = map_npz["times"]
      snaps  = map_npz["snaps"]

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

    # Compute song position for encoder conditioning
    total_duration_ms = mel.shape[0] * self.hop_ms_actual
    if total_duration_ms > 0:
      start_frac = segment_start_ms / total_duration_ms
      end_frac = segment_end_ms / total_duration_ms
    else:
      start_frac = 0.0
      end_frac = 1.0
    song_position = torch.tensor([start_frac, end_frac], dtype=torch.float32)

    window_tokens, loss_mask = self.token_window_builder.build(
      tokens=tokens,
      times=times,
      snaps=snaps,
      audio_start_ms=segment_start_ms,
      audio_end_ms=segment_end_ms,
    )

    return (
      torch.from_numpy(segment).float(),
      torch.from_numpy(window_tokens).long(),
      torch.from_numpy(loss_mask),
      song_position,
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
    self._sr_style_ids = np.array(
      [
        idx for tok, idx in tokenizer.token_to_id.items()
        if tok.startswith("SR_") or tok.startswith("STYLE_")
      ],
      dtype=np.int64,
    )

    # Precompute TS and SNAP token IDs for insertion
    self._ts_prefix = "TS_"
    self._snap_prefix = "SNAP_"
    self._spos_prefix = "SPOS_"
    self._dt_bin_ms = 10  # default, will be overridden if tokenizer has config
    if hasattr(tokenizer, 'config') and tokenizer.config is not None:
      self._dt_bin_ms = tokenizer.config.DT_BIN_MS
      self._ts_max_steps = tokenizer.config.TS_MAX_STEPS
      self._spos_bins = tokenizer.config.SPOS_BINS
    else:
      self._ts_max_steps = 1000
      self._spos_bins = 100

  def build(
    self,
    *,
    tokens: np.ndarray,            # (N,)
    times: np.ndarray,             # (N,) ms
    snaps: np.ndarray,             # (N,) snap subdivision values
    audio_start_ms: int,
    audio_end_ms: int,
  ) -> Tuple[np.ndarray, np.ndarray]:

    window_ms = audio_end_ms - audio_start_ms
    predict_start_ms = audio_start_ms + int(window_ms * self.overlap_ratio)

    # Find global tokens (before MAP_START)
    map_start_indices = np.where(tokens == self.MAP_START)[0]
    if len(map_start_indices) > 0:
      map_start_idx = map_start_indices[0]
      global_tokens = tokens[:map_start_idx]
    else:
      global_tokens = np.empty((0,), dtype=np.int64)

    # Song position token
    total_duration_ms = max(float(times[-1]), 1.0) if len(times) > 0 else 1.0
    spos_pct = int(audio_start_ms / total_duration_ms * 100)
    spos_pct = min(max(spos_pct, 0), self._spos_bins - 1)
    spos_token_str = f"{self._spos_prefix}{spos_pct}"
    spos_token_id = self.tokenizer.token_to_id.get(spos_token_str, None)

    # Get content tokens (after MAP_START, excluding structural tokens)
    content_mask = np.ones(len(tokens), dtype=bool)
    # Exclude global prefix and structural tokens
    if len(map_start_indices) > 0:
      content_mask[:map_start_idx + 1] = False  # exclude everything up to and including MAP_START
    content_mask &= (tokens != self.MAP_END)
    content_mask &= (tokens != self.EOS)
    if self._sr_style_ids.size:
      content_mask &= ~np.isin(tokens, self._sr_style_ids)

    content_tokens = tokens[content_mask]
    content_times = times[content_mask]
    content_snaps = snaps[content_mask]

    # Split into context (before predict_start) and target (predict_start to window_end)
    context_mask = content_times < predict_start_ms
    target_mask = (content_times >= predict_start_ms) & (content_times < audio_end_ms)

    context_struct_tokens = content_tokens[context_mask]
    context_struct_times = content_times[context_mask]
    context_struct_snaps = content_snaps[context_mask]

    target_struct_tokens = content_tokens[target_mask]
    target_struct_times = content_times[target_mask]
    target_struct_snaps = content_snaps[target_mask]

    # Insert TS/SNAP tokens before each new object/BPM timing
    context_with_ts = self._insert_ts_snap_tokens(
      context_struct_tokens, context_struct_times, context_struct_snaps, audio_start_ms
    )
    target_with_ts = self._insert_ts_snap_tokens(
      target_struct_tokens, target_struct_times, target_struct_snaps, audio_start_ms
    )

    # --- token budget ---
    prefix_parts = list(global_tokens)
    if spos_token_id is not None:
      prefix_parts.append(spos_token_id)
    prefix_parts.append(self.MAP_START)
    prefix_tokens = np.array(prefix_parts, dtype=np.int64)

    reserved = len(prefix_tokens) + len(target_with_ts)
    if reserved > self.max_tokens:
      target_with_ts = target_with_ts[-(self.max_tokens - len(prefix_tokens)):]
      reserved = len(prefix_tokens) + len(target_with_ts)

    available_context = self.max_tokens - reserved
    if available_context > 0 and len(context_with_ts) > available_context:
      context_with_ts = context_with_ts[-available_context:]
    elif available_context <= 0:
      context_with_ts = np.empty((0,), dtype=np.int64)

    window_tokens = np.concatenate([
      prefix_tokens,
      context_with_ts,
      target_with_ts,
    ])

    # Add MAP_END + EOS if the target contains MAP_END equivalent
    if self.MAP_END in target_struct_tokens:
      window_tokens = np.concatenate([
        window_tokens,
        np.array([self.MAP_END, self.EOS], dtype=np.int64),
      ])

    loss_mask = np.zeros(len(window_tokens), dtype=bool)
    target_start = len(prefix_tokens) + len(context_with_ts)
    loss_mask[target_start:] = True

    return window_tokens, loss_mask

  def _insert_ts_snap_tokens(
    self,
    tokens: np.ndarray,
    times: np.ndarray,
    snaps: np.ndarray,
    window_start_ms: int,
  ) -> np.ndarray:
    """Insert TS and SNAP tokens before each object boundary (OBJ_START or BPM token).

    The TS/SNAP pair is inserted before tokens that mark the start of a new
    temporal event: OBJ_START tokens and BPM tokens.
    """
    if len(tokens) == 0:
      return np.empty((0,), dtype=np.int64)

    obj_start_id = self.tokenizer.token_to_id["OBJ_START"]
    bpm_ids = {
      tid for tok, tid in self.tokenizer.token_to_id.items()
      if tok.startswith("BPM_")
    }
    trigger_ids = {obj_start_id} | bpm_ids

    result: list[int] = []

    for i, (tok_id, tok_time, tok_snap) in enumerate(zip(tokens, times, snaps)):
      if tok_id in trigger_ids:
        # Compute window-relative TS value
        ts_value = int(round((tok_time - window_start_ms) / self._dt_bin_ms))
        ts_value = max(0, min(ts_value, self._ts_max_steps - 1))

        ts_token_str = f"{self._ts_prefix}{ts_value}"
        ts_token_id = self.tokenizer.token_to_id.get(ts_token_str)

        snap_value = int(tok_snap)
        snap_value = max(0, min(snap_value, 16))
        snap_token_str = f"{self._snap_prefix}{snap_value}"
        snap_token_id = self.tokenizer.token_to_id.get(snap_token_str)

        if ts_token_id is not None:
          result.append(ts_token_id)
        if snap_token_id is not None:
          result.append(snap_token_id)

      result.append(tok_id)

    return np.array(result, dtype=np.int64)
