import click
import torch
from pathlib import Path
from tqdm.auto import tqdm
from .config import config_options, ExperimentConfig
from .osu import MapStyle
from .tokenizer import Tokenizer
from .model import build_model
from .data import audio_to_mel, normalize_mel
from .grammar import Grammar, GrammarState

@click.command()
@click.option("--audio", type=str, required=True, help="Path to input audio file")
@click.option("--output", type=str, help="Path to output generated file")
@click.option("--template", type=str, help="Path to template map file to base generation on")
@click.option("--bpm", type=str, required=True, help="Beats per minute for the generated file")
@click.option("--sr", type=int, default=4, help="Osu difficulty star rating target")
@click.option("--model", "checkpoint_path", type=str, required=True, help="Path to model checkpoint")
@click.option("--styles", type=str, help=f"Comma-separated list of styles to use for generation, {', '.join([s.value for s in MapStyle])}")
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature for generation")
@click.option("--max-len", type=int, default=1024, help="Maximum number of tokens to generate (including prefix)")
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
  strict: bool
):
  audio_path = Path(audio)
  if not audio_path.exists():
    raise FileNotFoundError(f"Audio file not found: {audio}")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokenizer = Tokenizer(config.tokenizer)
  style_tokens = _parse_styles(styles, tokenizer)
  bpm_value = _parse_bpm(bpm)
  prefix_tokens = _build_prefix_tokens(
    tokenizer=tokenizer,
    sr=sr,
    style_tokens=style_tokens,
    bpm=bpm_value,
  )

  mel, _ = audio_to_mel(
    path=audio_path,
    sample_rate=config.audio.sample_rate,
    hop_ms=config.audio.hop_ms,
    win_ms=config.audio.win_ms,
    n_mels=config.audio.n_mels,
    n_fft=config.audio.n_fft,
  )
  mel = mel.T  # (time, n_mels)
  if config.audio.normalize:
    mel = normalize_mel(mel, float(-37.2505), float(14.2582)) # Manual labour

  src = torch.from_numpy(mel).float().unsqueeze(0).to(device)

  model = build_model(config, vocab_size=len(tokenizer.vocab))
  model.to(device)
  model.eval()

  _load_checkpoint(model, checkpoint_path, device, strict=strict)

  with torch.no_grad():
    memory, memory_key_padding_mask = model.encoder(src, None)
    token_ids = _sample_decode(
      model=model,
      memory=memory,
      memory_key_padding_mask=memory_key_padding_mask,
      start_tokens=prefix_tokens,
      eos_id=tokenizer.vocab["EOS"],
      tokenizer=tokenizer,
      max_len=max_len,
      temperature=temperature,
    )

  readable_tokens = [tokenizer.id_to_token[t] for t in token_ids]
  generated_map = tokenizer.decode(token_ids)
  print(str(generated_map))


def _parse_bpm(bpm: str) -> float:
  try:
    return float(bpm)
  except ValueError as exc:
    raise ValueError(f"Invalid BPM value: {bpm}") from exc


def _parse_styles(styles: str | None, tokenizer: Tokenizer) -> list[str]:
  if not styles:
    return []
  names = [s.strip().upper() for s in styles.split(",") if s.strip()]
  style_tokens: list[str] = []
  for name in names:
    if name not in MapStyle.__members__:
      raise ValueError(f"Unknown style: {name}")
    token = f"STYLE_{name}"
    if token not in tokenizer.vocab:
      raise ValueError(f"Style token not in vocab: {token}")
    style_tokens.append(token)
  return style_tokens


def _build_prefix_tokens(
  *,
  tokenizer: Tokenizer,
  sr: int,
  style_tokens: list[str],
  bpm: float,
) -> list[int]:
  tokens: list[int] = []
  sr_token = tokenizer.find_closest_token_from_vocab(sr, "SR_")
  tokens.append(tokenizer.vocab[sr_token])
  for style_token in style_tokens:
    tokens.append(tokenizer.vocab[style_token])
  tokens.append(tokenizer.vocab["MAP_START"])
  if bpm is not None:
    bpm_token = tokenizer.find_closest_token_from_vocab(bpm, "BPM_")
    tokens.append(tokenizer.vocab["TP_START"])
    tokens.append(tokenizer.vocab[bpm_token])
    tokens.append(tokenizer.vocab["TP_END"])
  return tokens


def _load_checkpoint(model, checkpoint_path: str, device: torch.device, *, strict: bool) -> None:
  checkpoint = torch.load(checkpoint_path, map_location=device)
  state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
  if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
  try:
    model.load_state_dict(state_dict, strict=strict)
  except RuntimeError as exc:
    enc_layers = _count_layers(state_dict, "encoder.layers")
    dec_layers = _count_layers(state_dict, "decoder.layers")
    model_enc_layers = len(model.encoder.layers) if hasattr(model.encoder, "layers") else None
    model_dec_layers = len(model.decoder.layers) if hasattr(model.decoder, "layers") else None
    details = []
    if enc_layers is not None and model_enc_layers is not None:
      details.append(f"encoder layers: checkpoint={enc_layers}, model={model_enc_layers}")
    if dec_layers is not None and model_dec_layers is not None:
      details.append(f"decoder layers: checkpoint={dec_layers}, model={model_dec_layers}")
    hint = " | ".join(details) if details else "model/ckpt layer counts differ"
    raise RuntimeError(
      f"{exc}\nCheckpoint/config mismatch ({hint}). "
      "Run with the same --config/--size used for training, or pass --no-strict to ignore."
    ) from exc


def _count_layers(state_dict: dict[str, torch.Tensor], prefix: str) -> int | None:
  import re
  pattern = re.compile(rf"^{re.escape(prefix)}\\.(\\d+)\\.")
  indices = set()
  for key in state_dict.keys():
    match = pattern.match(key)
    if match:
      indices.add(int(match.group(1)))
  if not indices:
    return None
  return max(indices) + 1


def _sample_decode(
  *,
  model,
  memory: torch.Tensor,
  memory_key_padding_mask: torch.Tensor | None,
  start_tokens: list[int],
  eos_id: int,
  tokenizer: Tokenizer,
  max_len: int,
  temperature: float,
) -> list[int]:

  device = memory.device
  tokens = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

  grammar = Grammar(tokenizer=tokenizer)
  state = grammar.initial_state()

  for tid in start_tokens:
    state = grammar.consume(state, tid)

  steps = max_len - tokens.size(1)

  progress = tqdm(
    range(steps),
    desc="Generating",
    total=steps,
    dynamic_ncols=True,
  )

  for _ in progress:
    logits = model.decoder(
      tokens,
      memory,
      tgt_key_padding_mask=None,
      memory_key_padding_mask=memory_key_padding_mask,
      use_causal_mask=True,
    )

    next_logits = logits[:, -1, :]

    # --- GRAMMAR MASK ---
    allowed_ids = grammar.allowed_next_tokens(state)
    mask = torch.full_like(next_logits, float("-inf"))
    mask[:, list(allowed_ids)] = 0.0
    next_logits = next_logits + mask

    # --- SAMPLE ---
    if temperature <= 0:
      next_token = next_logits.argmax(dim=-1, keepdim=True)
    else:
      probs = torch.softmax(next_logits / temperature, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)

    tid = int(next_token.item())
    tokens = torch.cat([tokens, next_token], dim=1)

    # --- UPDATE GRAMMAR ---
    grammar.consume(state, tid)

    if tid == eos_id:
      break

  return tokens.squeeze(0).tolist()

if __name__ == "__main__":
  main()
