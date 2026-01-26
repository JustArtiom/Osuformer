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

@click.command()
@click.option("--audio", type=str, required=True, help="Path to input audio file")
@click.option("--output", type=str, help="Path to output generated file (unused)")
@click.option("--template", type=str, help="Path to template map file to base generation on (unused)")
@click.option("--bpm", type=str, required=True, help="Beats per minute for the generated file")
@click.option("--sr", type=int, default=4, help="Osu difficulty star rating target")
@click.option("--model", "checkpoint_path", type=str, required=True, help="Path to model checkpoint")
@click.option("--styles", type=str, help=f"Comma-separated list of styles to use for generation, {', '.join([s.value for s in MapStyle])}")
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature for generation")
@click.option("--max-len", type=int, default=4096, help="Maximum total number of tokens to generate")
@click.option("--start-ms", type=int, default=0, help="Start time in milliseconds (trim audio before generation)")
@click.option("--end-ms", type=int, default=None, help="End time in milliseconds (trim audio after generation)")
@click.option("--strict/--no-strict", default=True, help="Strictly enforce checkpoint compatibility")
@config_options
def main(
  config: ExperimentConfig,
  audio: str,
  output: str,
  template: str,
  bpm: str,
  sr: int,
  checkpoint_path: str,
  styles: str,
  temperature: float,
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

  dt_ms_by_id = _build_dt_ms_by_id(tokenizer)

  model = build_model(config, vocab_size=len(tokenizer.vocab))
  model.to(device)
  model.eval()
  _load_state_dict(model, state_dict, strict=strict)

  global_tokens = _build_global_tokens(tokenizer, sr=sr, styles=styles)
  initial_tokens = _build_initial_tokens(tokenizer, global_tokens=global_tokens, bpm=bpm)

  all_tokens: list[int] = []
  all_times: list[int] = []
  current_time_ms = 0
  for tid in initial_tokens:
    token_time, current_time_ms = _advance_time(tokenizer, tid, current_time_ms, dt_ms_by_id)
    all_tokens.append(tid)
    all_times.append(token_time)

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

  window_starts = _build_window_starts(mel.shape[0], segment_frames, hop_frames)
  if not window_starts:
    window_starts = [0]

  finished = False
  progress = tqdm(window_starts, desc="windows", dynamic_ncols=True)
  for start in progress:
    if finished or len(all_tokens) >= max_len:
      break

    segment = mel[start : start + segment_frames]
    if segment.shape[0] < segment_frames:
      pad_len = segment_frames - segment.shape[0]
      segment = _pad_segment(segment, pad_len)

    src = torch.from_numpy(segment).float().unsqueeze(0).to(device)
    with torch.no_grad():
      memory, memory_key_padding_mask = model.encoder(src, None)

    window_start_ms = int(round(start * hop_ms_actual))
    window_end_ms = int(round(window_start_ms + (segment_frames * hop_ms_actual)))
    predict_start_ms = window_start_ms + int(round((window_end_ms - window_start_ms) * config.dataset.overlap))

    context_tokens = _build_context_tokens(
      tokens=all_tokens,
      times=all_times,
      predict_start_ms=predict_start_ms,
      tokenizer=tokenizer,
      global_tokens=global_tokens,
      max_tokens=256,
    )

    prefix_tokens = list(global_tokens) + [tokenizer.vocab[Tok.MAP_START]] + context_tokens

    max_new_tokens = max_len - len(all_tokens)
    new_tokens, new_times, current_time_ms, finished = _generate_window_tokens(
      model=model,
      memory=memory,
      memory_key_padding_mask=memory_key_padding_mask,
      tokenizer=tokenizer,
      dt_ms_by_id=dt_ms_by_id,
      prefix_tokens=prefix_tokens,
      current_time_ms=current_time_ms,
      window_end_ms=window_end_ms,
      temperature=temperature,
      max_new_tokens=max_new_tokens,
    )

    all_tokens.extend(new_tokens)
    all_times.extend(new_times)

  pretty_tokens_printer([tokenizer.id_to_token[t] for t in all_tokens])

  print(tokenizer.decode(all_tokens))
  print("Generation complete.")

def pretty_tokens_printer(tokens: list[str]) -> None:
  def flush_dt(buffer: list[str]):
    if buffer:
      print(" ".join(buffer))
      buffer.clear()

  dt_buffer: list[str] = []
  i = 0

  while i < len(tokens):
    tok = tokens[i]

    if tok == "MAP_START" or tok == "MAP_END":
      flush_dt(dt_buffer)
      print(f"\n{tok}\n")

    # Timing points
    elif tok == "TP_START":
      flush_dt(dt_buffer)
      block = [tok]
      i += 1
      while i < len(tokens) and tokens[i] != "TP_END":
        block.append(tokens[i])
        i += 1
      block.append("TP_END")
      print(" ".join(block))

    # Objects
    elif tok == "OBJ_START":
      flush_dt(dt_buffer)
      block = [tok]
      i += 1
      while i < len(tokens) and tokens[i] != "OBJ_END":
        block.append(tokens[i])
        i += 1
      block.append("OBJ_END")
      print(" ".join(block))

    # Time deltas
    elif tok.startswith("DT_"):
      dt_buffer.append(tok)

    else:
      flush_dt(dt_buffer)
      print(tok)

    i += 1

  flush_dt(dt_buffer)

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


def _parse_bpm(bpm: str) -> float:
  try:
    return float(bpm)
  except ValueError as exc:
    raise ValueError(f"Invalid BPM value: {bpm}") from exc


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


def _build_initial_tokens(tokenizer: Tokenizer, *, global_tokens: list[int], bpm: str) -> list[int]:
  tokens = list(global_tokens)
  tokens.append(tokenizer.vocab[Tok.MAP_START])
  bpm_value = _parse_bpm(bpm)
  bpm_token = tokenizer.find_closest_token_from_vocab(bpm_value, Tok.BPM)
  tokens.append(tokenizer.vocab[Tok.TP_START])
  tokens.append(tokenizer.vocab[bpm_token])
  tokens.append(tokenizer.vocab[Tok.TP_END])
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


def _strip_structure(tokens: list[int], tokenizer: Tokenizer) -> list[int]:
  map_start = tokenizer.vocab[Tok.MAP_START]
  map_end = tokenizer.vocab[Tok.MAP_END]
  eos = tokenizer.vocab[Tok.EOS]
  sr_ids = [idx for tok, idx in tokenizer.vocab.items() if tok.startswith(Tok.SR)]
  style_ids = [idx for tok, idx in tokenizer.vocab.items() if tok.startswith(Tok.STYLE)]
  blocked = {map_start, map_end, eos} | set(sr_ids) | set(style_ids)
  return [t for t in tokens if t not in blocked]


def _build_context_tokens(
  *,
  tokens: list[int],
  times: list[int],
  predict_start_ms: int,
  tokenizer: Tokenizer,
  global_tokens: list[int],
  max_tokens: int,
) -> list[int]:
  context = [t for t, tm in zip(tokens, times) if tm < predict_start_ms]
  context = _strip_structure(context, tokenizer)

  reserved = len(global_tokens) + 1
  available = max_tokens - reserved
  if available <= 0:
    return []
  if len(context) > available:
    context = context[-available:]
  return context


def _build_dt_ms_by_id(tokenizer: Tokenizer) -> list[int]:
  """Precompute DT token deltas per token id.

  This avoids repeated string parsing/regex work in the tight generation loop.
  """
  dt_ms_by_id = [0 for _ in range(len(tokenizer.id_to_token))]
  for tid, tok in enumerate(tokenizer.id_to_token):
    if tok.startswith(Tok.DT):
      delta = tokenizer.extract_number_from_token(tok, Tok.DT)
      if delta is not None:
        dt_ms_by_id[tid] = int(delta)
  return dt_ms_by_id


def _advance_time(
  tokenizer: Tokenizer,
  token_id: int,
  current_time_ms: int,
  dt_ms_by_id: Sequence[int],
) -> tuple[int, int]:
  token_time = current_time_ms
  # Hot path: avoid string parsing on every token by using a precomputed lookup.
  delta = int(dt_ms_by_id[token_id])
  if delta:
    current_time_ms += delta
  return token_time, current_time_ms


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
  dt_ms_by_id: Sequence[int],
  prefix_tokens: list[int],
  current_time_ms: int,
  window_end_ms: int,
  temperature: float,
  max_new_tokens: int,
) -> tuple[list[int], list[int], int, bool]:
  grammar = Grammar(tokenizer=tokenizer)
  state = grammar.initial_state()

  cache = model.decoder.init_kv_cache(bsz=memory.size(0), device=memory.device, dtype=memory.dtype)
  last_logits: Optional[torch.Tensor] = None
  position = 0

  prefix_tokens, state = _sanitize_prefix_tokens(prefix_tokens, grammar, state)

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
    new_times: list[int] = []
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

      if temperature <= 0:
        next_token = next_logits.argmax(dim=-1, keepdim=True)
      else:
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

      tid = int(next_token.item())
      token_time, current_time_ms = _advance_time(tokenizer, tid, current_time_ms, dt_ms_by_id)
      new_tokens.append(tid)
      new_times.append(token_time)

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

  return new_tokens, new_times, current_time_ms, finished


if __name__ == "__main__":
  main()
