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


def train_one_epoch(model, loader, optimizer, device, epoch, sampler=None, scaler: Optional[GradScaler] = None, amp_dtype: Optional[torch.dtype] = None):
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

  for mel, tokens, loss_mask, token_pad_mask in progress:
    mel = mel.to(device, non_blocking=True)
    tokens = tokens.to(device, non_blocking=True)
    loss_mask = loss_mask.to(device, non_blocking=True)
    token_pad_mask = token_pad_mask.to(device, non_blocking=True)

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
        )

        loss = F.cross_entropy(
          logits.reshape(-1, logits.size(-1)),
          tokens_out.reshape(-1),
          reduction="none",
        )

        loss = loss.view(tokens_out.shape)  # (B, L-1)
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
      )

      loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tokens_out.reshape(-1),
        reduction="none",
      )

      loss = loss.view(tokens_out.shape)  # (B, L-1)
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
def validate_one_epoch(model, epoch, loader, device):
  model.eval()
  total_loss = 0.0
  total_tokens = 0

  progress = tqdm(
    loader,
    desc=f"val - Epoch {epoch}",
    leave=False,
    dynamic_ncols=True,
  )

  for mel, tokens, loss_mask, token_pad_mask in progress:
    mel = mel.to(device, non_blocking=True)
    tokens = tokens.to(device, non_blocking=True)
    loss_mask = loss_mask.to(device, non_blocking=True)
    token_pad_mask = token_pad_mask.to(device, non_blocking=True)

    tokens_in  = tokens[:, :-1]
    tokens_out = tokens[:, 1:]

    pad_mask  = token_pad_mask[:, :-1]
    loss_mask = loss_mask[:, 1:].float()

    logits = model(
      src=mel,
      tgt_tokens=tokens_in,
      tgt_key_padding_mask=pad_mask,
    )

    loss = F.cross_entropy(
      logits.reshape(-1, logits.size(-1)),
      tokens_out.reshape(-1),
      reduction="none",
    )

    loss = loss.view(tokens_out.shape)
    loss = loss * loss_mask
    progress.set_postfix(loss=f"{(loss.sum() / max(loss_mask.sum(), 1)).item():.4f}")

    total_loss += loss.sum().item()
    total_tokens += loss_mask.sum().item()

  return total_loss / max(total_tokens, 1)

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
    )

    val_loss = validate_one_epoch(model, epoch, val_loader, device)
    if distributed:
      val_loss_tensor = torch.tensor(val_loss, device=device, dtype=torch.float32)
      dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
      val_loss = val_loss_tensor.item() / dist.get_world_size()
    stop_tensor = torch.tensor(0, device=device, dtype=torch.int)
    if (not distributed) or dist.get_rank() == 0:
      assert checkpoint is not None
      stop = checkpoint.step(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
      )

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