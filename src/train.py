import os
import click
import torch
from pathlib import Path
from typing import Optional
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from .data import CachedDataset
from .config import config_options, ExperimentConfig
from .model import build_model
from tqdm.auto import tqdm

def setup_distributed():
  if "RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return True
  return False

def create_dataloader(dataset: CachedDataset, batch_size: int, workers: int = 1):
    sampler = None
    if dist.is_initialized():
      sampler = DistributedSampler(dataset, shuffle=True)

    loader = DataLoader(
      dataset,
      batch_size=batch_size,
      sampler=sampler,
      shuffle=(sampler is None),
      num_workers=workers,
      pin_memory=True,
      collate_fn=CachedDataset.collate_batch,
      persistent_workers=True,
    )

    return loader, sampler


def train_one_epoch(model, loader, optimizer, device, epoch, sampler=None):
  if sampler is not None:
    sampler.set_epoch(epoch)

  model.train()
  total_loss = 0.0

  progress = tqdm(
    loader,
    desc=f"Epoch {epoch}",
    leave=False,
    dynamic_ncols=True,
  )

  for mel, tokens, token_pad_mask in progress:
    mel = mel.to(device, non_blocking=True)
    tokens = tokens.to(device, non_blocking=True)
    token_pad_mask = token_pad_mask.to(device, non_blocking=True)

    tokens_in  = tokens[:, :-1]
    tokens_out = tokens[:, 1:]
    tgt_mask   = token_pad_mask[:, :-1]

    logits = model(
        src=mel,
        tgt_tokens=tokens_in,
        tgt_key_padding_mask=tgt_mask,
    )

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tokens_out.reshape(-1),
        ignore_index=0,
    )

    progress.set_postfix(loss=f"{loss.item():.4f}")
    total_loss += loss.item()
    loss.backward()
    optimizer.step()

  return total_loss / len(loader)


@click.command()
@click.option("--cache", "cache_name", type=str, required=True, help="Name of the cache to use for training")
@click.option("--batch-size", type=int, help="Batch size for training")
@click.option("--lr", type=float, help="Learning rate for optimizer")
@click.option("--workers", type=int, help="Number of data loading workers")
@click.option("--use-ram/--no-use-ram", help="Whether to load the entire cache into RAM")
@click.option("--epochs", type=int, help="Number of training epochs")
@config_options
def main(
  config: ExperimentConfig, 
  cache_name: str, 
  batch_size: int, 
  epochs: int, 
  lr: float, 
  workers: int, 
  use_ram: Optional[bool]
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

  train_dataset = CachedDataset(
    parent_path=Path(config.cache.path) / cache_name,
    window_ms=config.dataset.window_ms,
    hop_ms=config.audio.hop_ms,
    overlap=config.dataset.overlap,
    split="train"
  )

  loader, sampler = create_dataloader(
    dataset=train_dataset,
    batch_size=batch_size,
    workers=workers,
  )

  model = build_model(config, vocab_size=len(train_dataset.tokenizer.vocab))
  model.to(device)

  if distributed:
      model = torch.nn.parallel.DistributedDataParallel(
          model,
          device_ids=[torch.cuda.current_device()],
      )

  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

  for epoch in range(epochs):
    loss = train_one_epoch(
      model,
      loader,
      optimizer,
      device,
      epoch,
      sampler,
    )

    if not distributed or dist.get_rank() == 0:
      print(f"[Epoch {epoch}] loss = {loss:.4f}")

  if distributed:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()