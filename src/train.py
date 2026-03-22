import os
import click
import torch
from pathlib import Path
from typing import Optional
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from .data import CachedDataset, StreamingAudioStats
from .config import config_options, ExperimentConfig
from .model import build_model
from tqdm.auto import tqdm
from .checkpoint import Checkpoint

TS_LOSS_WEIGHT = 3.0
SNAP_LOSS_WEIGHT = 1.5
SL_LOSS_WEIGHT = 2.0

def build_token_weights(tokenizer, device: torch.device) -> torch.Tensor:
  vocab_size = len(tokenizer.vocab)
  weights = torch.ones(vocab_size, device=device, dtype=torch.float32)
  for tok, idx in tokenizer.token_to_id.items():
    if tok.startswith("TS_"):
      weights[idx] = TS_LOSS_WEIGHT
    elif tok.startswith("SNAP_"):
      weights[idx] = SNAP_LOSS_WEIGHT
    elif tok.startswith("SL_"):
      weights[idx] = SL_LOSS_WEIGHT
  return weights


def build_token_family_masks(tokenizer, device: torch.device) -> dict:
  """Precompute boolean masks for each token family for efficient accuracy computation."""
  vocab_size = len(tokenizer.vocab)
  families = {
    "ts": torch.zeros(vocab_size, dtype=torch.bool, device=device),
    "snap": torch.zeros(vocab_size, dtype=torch.bool, device=device),
    "position": torch.zeros(vocab_size, dtype=torch.bool, device=device),
    "type": torch.zeros(vocab_size, dtype=torch.bool, device=device),
    "timing": torch.zeros(vocab_size, dtype=torch.bool, device=device),
    "structure": torch.zeros(vocab_size, dtype=torch.bool, device=device),
  }
  ts_values = {}

  for tok, idx in tokenizer.token_to_id.items():
    if tok.startswith("TS_"):
      families["ts"][idx] = True
      families["timing"][idx] = True
      try:
        ts_values[idx] = int(float(tok[3:]))
      except ValueError:
        pass
    elif tok.startswith("SNAP_"):
      families["snap"][idx] = True
      families["timing"][idx] = True
    elif tok.startswith("X_") or tok.startswith("Y_"):
      families["position"][idx] = True
    elif tok in ("T_CIRCLE", "T_SLIDER", "T_SPINNER"):
      families["type"][idx] = True
    elif tok in ("OBJ_START", "OBJ_END", "MAP_START", "MAP_END", "EOS", "BOS"):
      families["structure"][idx] = True

  return families, ts_values


def compute_token_accuracies(
  logits: torch.Tensor,
  targets: torch.Tensor,
  loss_mask: torch.Tensor,
  families: dict,
  ts_values: dict,
) -> dict:
  """Compute per-token-family accuracies for monitoring training progress.

  Returns dict with keys like 'ts_accuracy', 'snap_accuracy', etc.
  """
  preds = logits.argmax(dim=-1)  # (B, L)
  correct = (preds == targets)    # (B, L)
  masked_correct = correct & loss_mask  # only count target tokens

  result = {}

  # Overall accuracy
  total = loss_mask.sum().item()
  if total > 0:
    result["overall_accuracy"] = masked_correct.sum().item() / total

  # Per-family accuracy
  for name, mask in families.items():
    family_mask = mask[targets] & loss_mask  # (B, L)
    family_total = family_mask.sum().item()
    if family_total > 0:
      family_correct = (masked_correct & family_mask).sum().item()
      result[f"{name}_accuracy"] = family_correct / family_total

  # Fuzzy TS accuracy (±1 and ±2 tolerance = ±10ms and ±20ms)
  ts_mask = families["ts"][targets] & loss_mask
  ts_total = ts_mask.sum().item()
  if ts_total > 0 and ts_values:
    # Build value tensors for targets and predictions
    device = targets.device
    target_vals = torch.zeros_like(targets, dtype=torch.float)
    pred_vals = torch.zeros_like(preds, dtype=torch.float)
    for tid, val in ts_values.items():
      target_vals[targets == tid] = val
      pred_vals[preds == tid] = val

    diff = (pred_vals - target_vals).abs()
    fuzzy_1 = ((diff <= 1) & ts_mask).sum().item()
    fuzzy_2 = ((diff <= 2) & ts_mask).sum().item()
    result["ts_fuzzy_1_accuracy"] = fuzzy_1 / ts_total
    result["ts_fuzzy_2_accuracy"] = fuzzy_2 / ts_total

  return result


def compute_loss_decomposition(
  logits: torch.Tensor,
  targets: torch.Tensor,
  loss_mask: torch.Tensor,
  families: dict,
) -> dict:
  """Compute loss decomposed by token family."""
  per_token_loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
    reduction="none",
  ).view(targets.shape)

  result = {}
  for name, mask in families.items():
    family_mask = mask[targets] & loss_mask
    family_total = family_mask.float().sum().clamp(min=1)
    family_loss = (per_token_loss * family_mask.float()).sum() / family_total
    result[f"{name}_loss"] = family_loss.item()

  return result

def setup_distributed():
  if "RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return True
  return False

def create_dataloader(dataset: CachedDataset, batch_size: int, workers: int = 1, shuffle: bool = True):
    sampler = None
    if dist.is_initialized():
      sampler = DistributedSampler(dataset, shuffle=shuffle)

    loader = DataLoader(
      dataset,
      batch_size=batch_size,
      sampler=sampler,
      shuffle=(sampler is None),
      num_workers=workers,
      pin_memory=True,
      collate_fn=CachedDataset.collate_batch,
      persistent_workers=(workers > 0),
    )

    return loader, sampler


def train_one_epoch(model, loader, optimizer, device, epoch, sampler=None, scaler: Optional[GradScaler] = None, amp_dtype: Optional[torch.dtype] = None, token_weights: Optional[torch.Tensor] = None):
  if sampler is not None:
    sampler.set_epoch(epoch)

  model.train()
  total_loss = 0.0

  progress = tqdm(
    loader,
    desc=f"train - Epoch {epoch}",
    leave=False,
    dynamic_ncols=True,
  )

  for mel, tokens, loss_mask, token_pad_mask, song_position in progress:
    mel = mel.to(device, non_blocking=True)
    tokens = tokens.to(device, non_blocking=True)
    loss_mask = loss_mask.to(device, non_blocking=True)
    token_pad_mask = token_pad_mask.to(device, non_blocking=True)
    song_position = song_position.to(device, non_blocking=True)

    tokens_in  = tokens[:, :-1]
    tokens_out = tokens[:, 1:]

    pad_mask   = token_pad_mask[:, :-1]
    loss_mask  = loss_mask[:, 1:]

    use_amp = scaler is not None and device.type == "cuda"
    if amp_dtype is None:
      amp_dtype = torch.bfloat16

    optimizer.zero_grad(set_to_none=True)

    if use_amp:
      with autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(
          src=mel,
          tgt_tokens=tokens_in,
          tgt_key_padding_mask=pad_mask,
          conditioning=song_position,
        )

        loss = F.cross_entropy(
          logits.reshape(-1, logits.size(-1)),
          tokens_out.reshape(-1),
          reduction="none",
        )

        loss = loss.view(tokens_out.shape)  # (B, L-1)
        if token_weights is not None:
          loss = loss * token_weights[tokens_out]
        loss = loss * loss_mask.float()    # zero out non-learning tokens
        denom = loss_mask.float().sum().clamp(min=1)
        loss = loss.sum() / denom

      progress.set_postfix(loss=f"{loss.item():.4f}")
      total_loss += loss.item()

      assert scaler is not None  # for mypy
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      scaler.step(optimizer)
      scaler.update()
    else:
      logits = model(
        src=mel,
        tgt_tokens=tokens_in,
        tgt_key_padding_mask=pad_mask,
        conditioning=song_position,
      )

      loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tokens_out.reshape(-1),
        reduction="none",
      )

      loss = loss.view(tokens_out.shape)  # (B, L-1)
      if token_weights is not None:
        loss = loss * token_weights[tokens_out]
      loss = loss * loss_mask.float()
      denom = loss_mask.float().sum().clamp(min=1)
      loss = loss.sum() / denom

      progress.set_postfix(loss=f"{loss.item():.4f}")
      total_loss += loss.item()

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

  return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, epoch, loader, device, token_weights: Optional[torch.Tensor] = None, families: Optional[dict] = None, ts_values: Optional[dict] = None):
  model.eval()
  total_loss = 0.0
  total_tokens = 0

  # Accumulators for per-token accuracy across batches
  acc_sums: dict = {}
  acc_counts: dict = {}
  loss_decomp_sums: dict = {}
  loss_decomp_batches = 0

  progress = tqdm(
    loader,
    desc=f"val - Epoch {epoch}",
    leave=False,
    dynamic_ncols=True,
  )

  for mel, tokens, loss_mask, token_pad_mask, song_position in progress:
    mel = mel.to(device, non_blocking=True)
    tokens = tokens.to(device, non_blocking=True)
    loss_mask = loss_mask.to(device, non_blocking=True)
    token_pad_mask = token_pad_mask.to(device, non_blocking=True)
    song_position = song_position.to(device, non_blocking=True)

    tokens_in  = tokens[:, :-1]
    tokens_out = tokens[:, 1:]

    pad_mask  = token_pad_mask[:, :-1]
    lm = loss_mask[:, 1:]

    logits = model(
      src=mel,
      tgt_tokens=tokens_in,
      tgt_key_padding_mask=pad_mask,
      conditioning=song_position,
    )

    loss = F.cross_entropy(
      logits.reshape(-1, logits.size(-1)),
      tokens_out.reshape(-1),
      reduction="none",
    )

    loss = loss.view(tokens_out.shape)
    if token_weights is not None:
      loss = loss * token_weights[tokens_out]
    loss_masked = loss * lm.float()
    progress.set_postfix(loss=f"{(loss_masked.sum() / max(lm.float().sum(), 1)).item():.4f}")

    total_loss += loss_masked.sum().item()
    total_tokens += lm.float().sum().item()

    # Compute per-token accuracies
    if families is not None and ts_values is not None:
      batch_acc = compute_token_accuracies(logits, tokens_out, lm, families, ts_values)
      for k, v in batch_acc.items():
        acc_sums[k] = acc_sums.get(k, 0.0) + v
        acc_counts[k] = acc_counts.get(k, 0) + 1

      batch_decomp = compute_loss_decomposition(logits, tokens_out, lm, families)
      for k, v in batch_decomp.items():
        loss_decomp_sums[k] = loss_decomp_sums.get(k, 0.0) + v
      loss_decomp_batches += 1

  avg_loss = total_loss / max(total_tokens, 1)

  # Average accuracies across batches
  avg_accuracies = {k: v / acc_counts[k] for k, v in acc_sums.items()} if acc_sums else None
  avg_decomposition = {k: v / loss_decomp_batches for k, v in loss_decomp_sums.items()} if loss_decomp_sums else None

  return avg_loss, avg_accuracies, avg_decomposition

@click.command()
@click.option("--cache", "cache_name", type=str, required=True, help="Name of the cache to use for training")
@click.option("--batch-size", type=int, help="Batch size for training")
@click.option("--lr", type=float, help="Learning rate for optimizer")
@click.option("--workers", type=int, help="Number of data loading workers")
@click.option("--use-ram/--no-use-ram", default=None, help="Whether to load the entire cache into RAM")
@click.option("--epochs", type=int, help="Number of training epochs")
@click.option("--ckp", "checkpoint_name", type=str, help="Path to save checkpoints")
@config_options
def main(
  config: ExperimentConfig,
  cache_name: str,
  batch_size: int,
  epochs: int,
  lr: float,
  workers: int,
  use_ram: Optional[bool],
  checkpoint_name: Optional[str],
):
  if not batch_size:
    batch_size = config.training.batch_size
  if not epochs:
    epochs = config.training.epochs
  if not workers:
    workers = config.training.workers
  if not lr:
    lr = config.training.lr
  if use_ram is None:
    use_ram = config.training.use_ram

  distributed = setup_distributed()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  train_audio_stats = StreamingAudioStats()
  train_dataset = CachedDataset(
    parent_path=Path(config.cache.path) / cache_name,
    window_ms=config.dataset.window_ms,
    hop_ms=config.audio.hop_ms,
    sample_rate=config.audio.sample_rate,
    overlap=config.dataset.overlap,
    audioStats=train_audio_stats,
    use_ram=use_ram,
    split="train"
  )

  val_audio_stats   = StreamingAudioStats()
  val_dataset = CachedDataset(
    parent_path=Path(config.cache.path) / cache_name,
    window_ms=config.dataset.window_ms,
    hop_ms=config.audio.hop_ms,
    sample_rate=config.audio.sample_rate,
    overlap=config.dataset.overlap,
    audioStats=val_audio_stats,
    use_ram=use_ram,
    split="val"
  )

  global_audio_stats = StreamingAudioStats()
  global_audio_stats.merge(train_audio_stats)
  global_audio_stats.merge(val_audio_stats)

  mean, std = global_audio_stats.finalize()
  print(f"Computed audio stats: mean={mean:.4f}, std={std:.4f}")
  train_dataset.load_audio_stats(mean, std)
  val_dataset.load_audio_stats(mean, std)

  token_weights = build_token_weights(train_dataset.tokenizer, device)
  families, ts_values = build_token_family_masks(train_dataset.tokenizer, device)

  train_loader, train_sampler = create_dataloader(
    dataset=train_dataset,
    batch_size=batch_size,
    workers=workers,
  )

  val_loader, _ = create_dataloader(
    dataset=val_dataset,
    batch_size=batch_size,
    workers=workers,
    shuffle=False,
  )

  model = build_model(config, vocab_size=len(train_dataset.tokenizer.vocab))
  model.to(device)

  if distributed:
      model = torch.nn.parallel.DistributedDataParallel(
          model,
          device_ids=[torch.cuda.current_device()],
      )

  scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))
  amp_dtype = torch.bfloat16

  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=config.training.lr_scheduler.factor,
    patience=config.training.lr_scheduler.patience,
    threshold=config.training.lr_scheduler.threshold,
    min_lr=config.training.lr_scheduler.min_lr,
  )

  checkpoint = None
  if (not distributed) or dist.get_rank() == 0:
    checkpoint = Checkpoint(
      config=config,
      tokenizer=train_dataset.tokenizer,
      name=checkpoint_name,
      mean=mean,
      std=std
    )
  for epoch in range(epochs):
    train_loss = train_one_epoch(
      model,
      train_loader,
      optimizer,
      device,
      epoch,
      train_sampler,
      scaler=scaler,
      amp_dtype=amp_dtype,
      token_weights=token_weights,
    )

    val_loss, token_accuracies, loss_decomposition = validate_one_epoch(
      model, epoch, val_loader, device,
      token_weights=token_weights,
      families=families,
      ts_values=ts_values,
    )
    if distributed:
      val_loss_tensor = torch.tensor(val_loss, device=device, dtype=torch.float32)
      dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
      val_loss = val_loss_tensor.item() / dist.get_world_size()
    scheduler.step(val_loss)
    stop_tensor = torch.tensor(0, device=device, dtype=torch.int)

    if (not distributed) or dist.get_rank() == 0:
      assert checkpoint is not None
      current_lr = optimizer.param_groups[0]["lr"]
      stop = checkpoint.step(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        current_lr=current_lr,
        token_accuracies=token_accuracies,
        loss_decomposition=loss_decomposition,
      )

      print(f"[Epoch {epoch}] lr={current_lr:.2e}")
      print(f"[Epoch {epoch}] loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")
      stop_tensor.fill_(1 if stop else 0)
    if distributed:
      dist.broadcast(stop_tensor, src=0)
    if stop_tensor.item() == 1:
      if distributed:
        dist.barrier()
      if (not distributed) or dist.get_rank() == 0:
        print("Early stopping triggered.")
      break

  if distributed:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
