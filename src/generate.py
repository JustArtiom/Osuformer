import click
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import config_options, ExperimentConfig
from .constraints import Tok
from .data import audio_to_mel, normalize_mel, ms_to_samples
from .grammar import Grammar, GrammarState
from .model import build_model
from .osu import MapStyle
from .tokenizer import Tokenizer

DEFAULT_TIMING_TEMPERATURE = 0.1


@click.command()
@click.option("--audio", type=str, required=True, help="Path to input audio file")
@click.option("--output", type=str, help="Path to output generated file")
@click.option("--bpm", type=str, default=None, help="Beats per minute (optional, model predicts from audio if not given)")
@click.option("--offset", type=int, default=0, help="Offset in milliseconds")
@click.option("--sr", type=int, default=4, help="Osu difficulty star rating target")
@click.option("--model", "checkpoint_path", type=str, required=True, help="Path to model checkpoint")
@click.option("--styles", type=str, help=f"Comma-separated list of styles: {', '.join([s.value for s in MapStyle])}")
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature for generation")
@click.option("--timing-temperature", type=float, default=DEFAULT_TIMING_TEMPERATURE, help="Temperature for timing (TS/SNAP) tokens")
@click.option("--max-len", type=int, default=4096, help="Maximum total number of tokens to generate")
@click.option("--start-ms", type=int, default=0, help="Start time in milliseconds")
@click.option("--end-ms", type=int, default=None, help="End time in milliseconds")
@click.option("--strict/--no-strict", default=True, help="Strictly enforce checkpoint compatibility")
@config_options
def main(
  config: ExperimentConfig,
  audio: str,
  output: str,
  bpm: Optional[str],
  offset: int,
  sr: int,
  checkpoint_path: str,
  styles: str,
  temperature: float,
  timing_temperature: float,
  max_len: int,
  start_ms: int,
  end_ms: Optional[int],
  strict: bool,
):
  audio_path = Path(audio)
  if not audio_path.exists():
    raise FileNotFoundError(f"Audio file not found: {audio}")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  checkpoint, state_dict, mean, std, vocab = _load_checkpoint_assets(checkpoint_path, device)
  tokenizer = _tokenizer_from_vocab(config, vocab)

  model = build_model(config, vocab_size=len(tokenizer.vocab))
  model.to(device)
  model.eval()
  _load_state_dict(model, state_dict, strict=strict)

  # Build precomputed lookups
  ts_ids = _build_ts_groups(tokenizer)
  snap_ids = {tid for tok, tid in tokenizer.vocab.items() if tok.startswith(Tok.SNAP)}

  global_tokens = _build_global_tokens(tokenizer, sr=sr, styles=styles)

  mel, _ = audio_to_mel(
    path=audio_path,
    sample_rate=config.audio.sample_rate,
    hop_ms=config.audio.hop_ms,
    win_ms=config.audio.win_ms,
    n_mels=config.audio.n_mels,
    n_fft=config.audio.n_fft,
  )
  mel = mel.T  # (time, n_mels)

  mel = _slice_mel(mel, hop_ms=config.audio.hop_ms, start_ms=start_ms, end_ms=end_ms)

  mel = normalize_mel(mel, mean, std)

  segment_frames = config.dataset.window_ms // config.audio.hop_ms
  hop_frames = int(segment_frames * (1 - config.dataset.overlap))
  hop_samples = ms_to_samples(config.audio.sample_rate, config.audio.hop_ms)
  hop_ms_actual = hop_samples * 1000.0 / config.audio.sample_rate

  total_duration_ms = mel.shape[0] * hop_ms_actual

  window_starts = _build_window_starts(mel.shape[0], segment_frames, hop_frames)
  if not window_starts:
    window_starts = [0]

  # Build initial tokens for the sequence
  initial_tokens = _build_initial_tokens(
    tokenizer, global_tokens=global_tokens,
    bpm=bpm, offset=offset,
    total_duration_ms=total_duration_ms,
    window_start_ms=0,
    dt_bin_ms=config.tokenizer.DT_BIN_MS,
    spos_bins=config.tokenizer.SPOS_BINS,
  )

  all_tokens: list[int] = list(initial_tokens)
  all_times: list[float] = [0.0] * len(initial_tokens)

  finished = False
  last_ts_value = -1

  progress = tqdm(window_starts, desc="windows", dynamic_ncols=True)
  for win_idx, start in enumerate(progress):
    if finished or len(all_tokens) >= max_len:
      break

    segment = mel[start : start + segment_frames]
    if segment.shape[0] < segment_frames:
      pad_len = segment_frames - segment.shape[0]
      segment = _pad_segment(segment, pad_len)

    src = torch.from_numpy(segment).float().unsqueeze(0).to(device)

    window_start_ms = int(round(start * hop_ms_actual))
    window_end_ms = int(round(window_start_ms + (segment_frames * hop_ms_actual)))

    # Compute song position conditioning
    start_frac = window_start_ms / max(total_duration_ms, 1.0)
    end_frac = window_end_ms / max(total_duration_ms, 1.0)
    conditioning = torch.tensor([[start_frac, end_frac]], device=device, dtype=torch.float32)

    with torch.no_grad():
      memory, memory_key_padding_mask = model.encoder(src, None, conditioning=conditioning)

    predict_start_ms = window_start_ms + int(round((window_end_ms - window_start_ms) * config.dataset.overlap))

    context_tokens = _build_context_tokens(
      tokens=all_tokens,
      times=all_times,
      predict_start_ms=predict_start_ms,
      window_start_ms=window_start_ms,
      tokenizer=tokenizer,
      global_tokens=global_tokens,
      max_tokens=1024,
      dt_bin_ms=config.tokenizer.DT_BIN_MS,
      ts_max_steps=config.tokenizer.TS_MAX_STEPS,
      spos_bins=config.tokenizer.SPOS_BINS,
      total_duration_ms=total_duration_ms,
    )

    # Build SPOS prefix token for this window
    spos_pct = int(window_start_ms / max(total_duration_ms, 1.0) * 100)
    spos_pct = min(max(spos_pct, 0), config.tokenizer.SPOS_BINS - 1)
    spos_token_str = f"{Tok.SPOS}{spos_pct}"
    spos_tid = tokenizer.vocab.get(spos_token_str)

    prefix_tokens = list(global_tokens)
    if spos_tid is not None:
      prefix_tokens.append(spos_tid)
    prefix_tokens.append(tokenizer.vocab[Tok.MAP_START])
    prefix_tokens.extend(context_tokens)

    max_new_tokens = max_len - len(all_tokens)
    new_tokens, new_times, last_ts_value, finished = _generate_window_tokens(
      model=model,
      memory=memory,
      memory_key_padding_mask=memory_key_padding_mask,
      tokenizer=tokenizer,
      ts_ids=ts_ids,
      snap_ids=snap_ids,
      prefix_tokens=prefix_tokens,
      window_start_ms=window_start_ms,
      window_end_ms=window_end_ms,
      last_ts_value=last_ts_value,
      temperature=temperature,
      timing_temperature=timing_temperature,
      max_new_tokens=max_new_tokens,
      dt_bin_ms=config.tokenizer.DT_BIN_MS,
    )

    all_tokens.extend(new_tokens)
    all_times.extend(new_times)

  pretty_tokens_printer([tokenizer.id_to_token[t] for t in all_tokens])

  beatmap = tokenizer.decode(all_tokens)
  print(beatmap)
  print("Generation complete.")


def pretty_tokens_printer(tokens: list[str]) -> None:
  def flush_buf(buffer: list[str]):
    if buffer:
      print(" ".join(buffer))
      buffer.clear()

  buf: list[str] = []
  i = 0

  while i < len(tokens):
    tok = tokens[i]

    if tok == "MAP_START" or tok == "MAP_END":
      flush_buf(buf)
      print(f"\n{tok}\n")

    elif tok.startswith("TS_") or tok.startswith("SNAP_"):
      buf.append(tok)

    elif tok == "OBJ_START":
      flush_buf(buf)
      block = [tok]
      i += 1
      while i < len(tokens) and tokens[i] != "OBJ_END":
        block.append(tokens[i])
        i += 1
      block.append("OBJ_END")
      print(" ".join(block))

    else:
      flush_buf(buf)
      print(tok)

    i += 1

  flush_buf(buf)


def _load_checkpoint_assets(checkpoint_path: str, device: torch.device):
  checkpoint = torch.load(checkpoint_path, map_location=device)
  state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
  if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

  if not isinstance(checkpoint, dict):
    raise ValueError("Checkpoint missing metadata (mean/std/vocab).")

  if "mean" not in checkpoint or "std" not in checkpoint:
    raise ValueError("Checkpoint missing mean/std for mel normalization.")
  if "vocab" not in checkpoint:
    raise ValueError("Checkpoint missing vocab for tokenizer.")

  mean = float(checkpoint["mean"])
  std = float(checkpoint["std"])
  vocab = checkpoint["vocab"]
  if not isinstance(vocab, dict):
    raise ValueError("Checkpoint vocab must be a dict of token -> id.")

  return checkpoint, state_dict, mean, std, vocab


def _tokenizer_from_vocab(config: ExperimentConfig, vocab: dict[str, int]) -> Tokenizer:
  tokenizer = Tokenizer(config.tokenizer)
  tokenizer.vocab = vocab
  tokenizer.token_to_id = vocab
  tokenizer.id_to_token = ["" for _ in range(len(vocab))]
  for token, idx in vocab.items():
    tokenizer.id_to_token[idx] = token
  return tokenizer


def _load_state_dict(model, state_dict: dict[str, torch.Tensor], *, strict: bool) -> None:
  try:
    model.load_state_dict(state_dict, strict=strict)
  except RuntimeError as exc:
    raise RuntimeError(
      f"Checkpoint/config mismatch. Run with the same --config/--size used for training, "
      f"or pass --no-strict to ignore.\n{exc}"
    ) from exc


def _parse_styles(styles: str | None) -> list[str]:
  if not styles:
    return []
  return [s.strip().upper() for s in styles.split(",") if s.strip()]


def _build_global_tokens(tokenizer: Tokenizer, *, sr: int, styles: str | None) -> list[int]:
  tokens: list[int] = []
  sr_token = tokenizer.find_closest_token_from_vocab(sr, Tok.SR)
  tokens.append(tokenizer.vocab[sr_token])

  for style in _parse_styles(styles):
    if style not in MapStyle.__members__:
      raise ValueError(f"Unknown style: {style}")
    style_token = f"{Tok.STYLE}{style}"
    if style_token not in tokenizer.vocab:
      raise ValueError(f"Style token not in vocab: {style_token}")
    tokens.append(tokenizer.vocab[style_token])
  return tokens


def _build_initial_tokens(
  tokenizer: Tokenizer,
  *,
  global_tokens: list[int],
  bpm: Optional[str],
  offset: int,
  total_duration_ms: float,
  window_start_ms: int,
  dt_bin_ms: int,
  spos_bins: int,
) -> list[int]:
  tokens = list(global_tokens)

  # Song position token for first window
  spos_pct = int(window_start_ms / max(total_duration_ms, 1.0) * 100)
  spos_pct = min(max(spos_pct, 0), spos_bins - 1)
  spos_token_str = f"{Tok.SPOS}{spos_pct}"
  spos_tid = tokenizer.vocab.get(spos_token_str)
  if spos_tid is not None:
    tokens.append(spos_tid)

  tokens.append(tokenizer.vocab[Tok.MAP_START])

  # If BPM is provided, insert TS_offset SNAP_0 BPM_n at the start
  if bpm is not None:
    bpm_value = float(bpm)
    bpm_token = tokenizer.find_closest_token_from_vocab(bpm_value, "BPM_")

    ts_value = max(0, int(round(offset / dt_bin_ms)))
    ts_token_str = f"{Tok.TS}{ts_value}"
    ts_tid = tokenizer.vocab.get(ts_token_str)
    snap_tid = tokenizer.vocab.get(f"{Tok.SNAP}0")

    if ts_tid is not None:
      tokens.append(ts_tid)
    if snap_tid is not None:
      tokens.append(snap_tid)
    tokens.append(tokenizer.vocab[bpm_token])

  return tokens


def _slice_mel(mel, *, hop_ms: int, start_ms: int, end_ms: Optional[int]):
  if start_ms < 0:
    raise ValueError("start-ms must be >= 0")
  if end_ms is not None and end_ms < start_ms:
    raise ValueError("end-ms must be >= start-ms")

  total_frames = mel.shape[0]
  start_frame = min(total_frames, start_ms // hop_ms)
  end_frame = total_frames if end_ms is None else min(total_frames, int(math.ceil(end_ms / hop_ms)))
  return mel[start_frame:end_frame]


def _build_window_starts(total_frames: int, segment_frames: int, hop_frames: int) -> list[int]:
  if total_frames <= 0:
    return []
  starts = list(range(0, max(total_frames - segment_frames + 1, 0), max(hop_frames, 1)))
  final_start = max(0, total_frames - segment_frames)
  if not starts or starts[-1] != final_start:
    starts.append(final_start)
  return starts


def _pad_segment(segment: np.ndarray, pad_len: int) -> np.ndarray:
  if pad_len <= 0:
    return segment
  return np.pad(
    segment,
    ((0, pad_len), (0, 0)),
    mode="constant",
    constant_values=0.0,
  )


def _build_ts_groups(tokenizer: Tokenizer) -> dict[int, int]:
  """Build a dict mapping TS token_id -> TS value (int) for fast lookup."""
  ts_groups: dict[int, int] = {}
  for tok, tid in tokenizer.vocab.items():
    if tok.startswith(Tok.TS):
      try:
        ts_groups[tid] = int(float(tok[len(Tok.TS):]))
      except ValueError:
        pass
  return ts_groups


def _strip_structure(tokens: list[int], tokenizer: Tokenizer) -> list[int]:
  map_start = tokenizer.vocab[Tok.MAP_START]
  map_end = tokenizer.vocab[Tok.MAP_END]
  eos = tokenizer.vocab[Tok.EOS]
  sr_ids = {idx for tok, idx in tokenizer.vocab.items() if tok.startswith(Tok.SR)}
  style_ids = {idx for tok, idx in tokenizer.vocab.items() if tok.startswith(Tok.STYLE)}
  spos_ids = {idx for tok, idx in tokenizer.vocab.items() if tok.startswith(Tok.SPOS)}
  blocked = {map_start, map_end, eos} | sr_ids | style_ids | spos_ids
  return [t for t in tokens if t not in blocked]


def _build_context_tokens(
  *,
  tokens: list[int],
  times: list[float],
  predict_start_ms: int,
  window_start_ms: int,
  tokenizer: Tokenizer,
  global_tokens: list[int],
  max_tokens: int,
  dt_bin_ms: int,
  ts_max_steps: int,
  spos_bins: int,
  total_duration_ms: float,
) -> list[int]:
  """Build context tokens for a window, re-anchoring TS values to window_start_ms."""
  # Collect tokens before predict_start
  context = []
  for t, tm in zip(tokens, times):
    if tm < predict_start_ms:
      context.append((t, tm))

  # Strip structural tokens
  stripped = [(t, tm) for t, tm in context
              if t not in _get_blocked_ids(tokenizer)]

  # Re-anchor: replace any TS tokens with window-relative values,
  # and rebuild TS/SNAP pairs for object-starting tokens
  reanchored: list[int] = []
  i = 0
  while i < len(stripped):
    t, tm = stripped[i]
    tok_str = tokenizer.id_to_token[t]

    if tok_str.startswith(Tok.TS):
      # Recompute TS value relative to current window
      new_ts = int(round((tm - window_start_ms) / dt_bin_ms))
      new_ts = max(0, min(new_ts, ts_max_steps - 1))
      new_ts_str = f"{Tok.TS}{new_ts}"
      new_ts_tid = tokenizer.vocab.get(new_ts_str)
      if new_ts_tid is not None:
        reanchored.append(new_ts_tid)
    else:
      reanchored.append(t)
    i += 1

  # Budget
  reserved = len(global_tokens) + 2  # SPOS + MAP_START
  available = max_tokens - reserved
  if available <= 0:
    return []
  if len(reanchored) > available:
    reanchored = reanchored[-available:]
  return reanchored


_blocked_ids_cache: dict[int, set[int]] = {}

def _get_blocked_ids(tokenizer: Tokenizer) -> set[int]:
  key = id(tokenizer)
  if key not in _blocked_ids_cache:
    map_start = tokenizer.vocab[Tok.MAP_START]
    map_end = tokenizer.vocab[Tok.MAP_END]
    eos = tokenizer.vocab[Tok.EOS]
    sr_ids = {idx for tok, idx in tokenizer.vocab.items() if tok.startswith(Tok.SR)}
    style_ids = {idx for tok, idx in tokenizer.vocab.items() if tok.startswith(Tok.STYLE)}
    spos_ids = {idx for tok, idx in tokenizer.vocab.items() if tok.startswith(Tok.SPOS)}
    _blocked_ids_cache[key] = {map_start, map_end, eos} | sr_ids | style_ids | spos_ids
  return _blocked_ids_cache[key]


def _sanitize_prefix_tokens(
  tokens: list[int],
  grammar: Grammar,
  state: GrammarState,
) -> tuple[list[int], GrammarState]:
  sanitized: list[int] = []
  for tid in tokens:
    allowed = grammar.allowed_next_tokens(state)
    if tid in allowed:
      sanitized.append(tid)
      state = grammar.update_state(state, tid)
  return sanitized, state


def _generate_window_tokens(
  *,
  model,
  memory: torch.Tensor,
  memory_key_padding_mask: torch.Tensor | None,
  tokenizer: Tokenizer,
  ts_ids: dict[int, int],
  snap_ids: set[int],
  prefix_tokens: list[int],
  window_start_ms: int,
  window_end_ms: int,
  last_ts_value: int,
  temperature: float,
  timing_temperature: float,
  max_new_tokens: int,
  dt_bin_ms: int,
) -> tuple[list[int], list[float], int, bool]:
  grammar = Grammar(tokenizer=tokenizer)
  state = grammar.initial_state()

  # Override grammar's last_ts_value for cross-window monotonic continuity
  state.last_ts_value = last_ts_value

  cache = model.decoder.init_kv_cache(bsz=memory.size(0), device=memory.device, dtype=memory.dtype)
  last_logits: Optional[torch.Tensor] = None
  position = 0

  prefix_tokens, state = _sanitize_prefix_tokens(prefix_tokens, grammar, state)

  current_time_ms = float(window_start_ms)

  with torch.no_grad():
    logits_process = tqdm(prefix_tokens, desc="Prefix", leave=False, dynamic_ncols=True)
    for tid in logits_process:
      token_tensor = torch.tensor([tid], device=memory.device)
      logits, cache = model.decoder.forward_step(
        token_tensor,
        memory,
        cache=cache,
        memory_key_padding_mask=memory_key_padding_mask,
        position=position,
      )
      last_logits = logits
      position += 1

    new_tokens: list[int] = []
    new_times: list[float] = []
    finished = False

    token_progress = tqdm(range(max_new_tokens), desc="Generating", leave=False, dynamic_ncols=True)
    for _ in token_progress:
      if last_logits is None:
        break
      if current_time_ms >= window_end_ms:
        break

      next_logits = last_logits[:, -1, :]
      allowed_ids = grammar.allowed_next_tokens(state)
      if allowed_ids:
        allowed_list = list(allowed_ids)
        mask = torch.full_like(next_logits, float("-inf"))
        mask[:, allowed_list] = 0.0
        next_logits = next_logits + mask

      # Determine if this is a timing token position
      is_timing = allowed_ids and allowed_ids.issubset(set(ts_ids.keys()) | snap_ids)
      effective_temp = timing_temperature if is_timing else temperature

      if effective_temp <= 0:
        next_token = next_logits.argmax(dim=-1, keepdim=True)
      else:
        probs = torch.softmax(next_logits / effective_temp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

      tid = int(next_token.item())

      # Update time tracking based on TS tokens
      if tid in ts_ids:
        ts_val = ts_ids[tid]
        current_time_ms = window_start_ms + ts_val * dt_bin_ms
        last_ts_value = ts_val

      new_tokens.append(tid)
      new_times.append(current_time_ms)

      state = grammar.consume(state, tid)

      if tid == tokenizer.vocab[Tok.EOS] or tid == tokenizer.vocab[Tok.MAP_END]:
        finished = True
        break

      logits, cache = model.decoder.forward_step(
        next_token,
        memory,
        cache=cache,
        memory_key_padding_mask=memory_key_padding_mask,
        position=position,
      )
      last_logits = logits
      position += 1

  return new_tokens, new_times, last_ts_value, finished


if __name__ == "__main__":
  main()
